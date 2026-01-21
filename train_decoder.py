import pickle
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime
from copy import deepcopy
import csv 

from neural_net import EVChargingDecoder
from utils import EVChargingDataPoint, load_saved_data_points
from imputer import EVChargingImputationDataset
from config import NUM_STATIONS
from rag import RAGIndex



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger("train_decoder")


def setup_metrics_csv(filename: str) -> str:
    """Setup CSV file for tracking metrics."""
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = [
            'epoch', 'train_loss', 'train_rec_loss', 'train_kl_loss', 
            'val_loss', 'val_rec_loss', 'val_kl_loss',
            'val_imputation_mse', 'val_imputation_rmse', 'val_imputation_mae', 'val_imputation_mape',
            'is_best'
        ]
        csv_writer.writerow(header)
        
    return filename


def update_metrics_csv(filename: str, epoch: int, train_metrics: dict, val_metrics: dict, is_best: bool):
    """Update CSV with metrics for the current epoch."""
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        row = [
            epoch, 
            train_metrics.get('loss', None), 
            train_metrics.get('rec_loss', None),
            train_metrics.get('kl_loss', None),
            val_metrics.get('val_loss', None),
            val_metrics.get('val_rec_loss', None),
            val_metrics.get('val_kl_loss', None),
            val_metrics.get('val_imputation_mse', None),
            val_metrics.get('val_imputation_rmse', None),
            val_metrics.get('val_imputation_mae', None),
            val_metrics.get('val_imputation_mape', None),
            'Yes' if is_best else 'No'
        ]
        csv_writer.writerow(row)




def prep_rag_train_decoder(data_points: List[EVChargingDataPoint]) -> RAGIndex:
    """
    Prepare training, validation, and test data with RAG index.
    
    Args:
        data_points: List of complete data points
    Returns:
        Tuple containing:
        - RAG index with complete training data
    """
    
    
    if len(data_points) == 0:
        raise ValueError("No complete data points with embeddings found")
    
    
    # Create RAG index with complete training data
    embedding_dim = data_points[0].embedding.shape[0]
    rag_index = RAGIndex(embedding_dim=embedding_dim)
    rag_index.batch_add_datapoints(data_points=data_points)
    
    
    logger.info(f"Created RAG index with {rag_index.get_size()} data points")
    

    return rag_index




def compute_masked_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    missing_mask: torch.Tensor,
    dataset_name: str,
    kl_weight: float=0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss with uncertainty weighting and KL divergence, focused on missing values.
    
    Args:
        outputs: Model outputs from forward pass
        targets: Target charging demand values [batch_size, seq_length, output_dim]
        missing_mask: Boolean mask indicating missing values [batch_size, seq_length, output_dim]
        kl_weight: Weight for KL divergence term
        
    Returns:
        Tuple containing:
        - Total loss
        - Reconstruction loss
        - KL loss
    """
    # Extract predictions and uncertainties
    mean = outputs["mean"]
    logvar = outputs["logvar"]
    
    # Focus on missing values (these are the ones we want to impute)
    mean_missing = mean[missing_mask]
    logvar_missing = logvar[missing_mask]
    if dataset_name in ['Dundee', 'Perth']:
        logvar_missing = torch.clamp(logvar_missing, -3, 3)
    targets_missing = targets[missing_mask]
    
    # If no missing values in this batch, use all values
    if mean_missing.numel() == 0:
        mean_missing = mean
        logvar_missing = logvar
        targets_missing = targets
    
    # Precision-weighted MSE on missing values
    precision = torch.exp(-logvar_missing)
    rec_loss = torch.mean(precision * (targets_missing - mean_missing)**2 + logvar_missing)
    
    # KL divergence for variational component
    mu, logvar_z = outputs["mu"], outputs["logvar_z"]
    kl_loss = -0.5 * torch.mean(1 + logvar_z - mu.pow(2) - logvar_z.exp())
    
    # Total loss
    total_loss = rec_loss + kl_weight * kl_loss
    
    return total_loss, rec_loss, kl_loss


def train_epoch(
    model: EVChargingDecoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataset_name: str,
    kl_weight: float=0.1,
    sample: bool=True,
) -> Dict[str, float]:
    """
    Train for one epoch using imputation-focused approach.
    
    Args:
        model: Model to train
        dataloader: DataLoader with training data containing missing values
        optimizer: Optimizer
        device: Device to train on
        kl_weight: Weight for KL divergence loss
        sample: Whether to sample from latent distribution
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    
    for batch in dataloader:
        # Move everything to device
        embedding = batch["embedding"].to(device)
        station_id = batch["station_id"].to(device)
        calendar_features = batch["calendar_features"].to(device)
        normalized_history = batch['normalized_history'].to(device)
        masked_normalized_history = batch['masked_normalized_history'].to(device)
        missing_mask = batch["missing_mask"].to(device)
        
        
        # Forward pass - using masked history as initial sequence
        outputs = model.forward(
            llm_embedding=embedding, 
            station_id=station_id, 
            calendar_features=calendar_features, 
            initial_sequence=masked_normalized_history,
            sample=sample, 
            mask=missing_mask
        )
        
        # Calculate loss focused on missing values
        loss, rec_loss, kl_loss = compute_masked_loss(
            outputs=outputs, 
            targets=normalized_history, 
            missing_mask=missing_mask, 
            dataset_name=dataset_name,
            kl_weight=kl_weight
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_kl_loss += kl_loss.item()
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_rec_loss = total_rec_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "kl_loss": avg_kl_loss
    }




def validate(
    model: EVChargingDecoder,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str,
    kl_weight: float=0.1,
) -> Dict[str, float]:
    """
    Validate model using imputation-focused approach.
    
    Args:
        model: Model to validate
        dataloader: DataLoader with validation data containing missing values
        device: Device to validate on
        kl_weight: Weight for KL divergence loss
        
    Returns:
        Dictionary with validation metrics including imputation metrics
    """
    model.eval()
    total_loss = 0
    total_rec_loss = 0
    total_kl_loss = 0
    
    # For imputation metrics
    all_targets = []
    all_predictions = []
    all_missing_masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move everything to device
            embedding = batch["embedding"].to(device)
            station_id = batch["station_id"].to(device)
            calendar_features = batch["calendar_features"].to(device)
            history = batch["history"].to(device)
            normalized_history = batch['normalized_history'].to(device)
            masked_normalized_history = batch['masked_normalized_history'].to(device)
            history_mean = batch['history_mean'].to(device)
            history_std = batch['history_std'].to(device)
            missing_mask = batch["missing_mask"].to(device)
            

            # Forward pass - using masked history
            outputs = model.forward(
                llm_embedding=embedding, 
                station_id=station_id, 
                calendar_features=calendar_features,
                initial_sequence=masked_normalized_history,
                sample=False,
                mask=missing_mask
            )
            
            # Calculate loss focused on missing values
            loss, rec_loss, kl_loss = compute_masked_loss(
                outputs=outputs, 
                targets=normalized_history, 
                missing_mask=missing_mask, 
                dataset_name=dataset_name,
                kl_weight=kl_weight
            )
            
            # Update loss metrics
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Get predictions and inverse-normalize
            predictions = outputs["mean"]
            predictions = predictions * history_std.unsqueeze(1).unsqueeze(1) + history_mean.unsqueeze(1).unsqueeze(1)
            

            # Store for metric calculation
            all_targets.append(history.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_missing_masks.append(missing_mask.cpu().numpy())
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_rec_loss = total_rec_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    # Calculate imputation metrics
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_missing_masks = np.concatenate(all_missing_masks, axis=0)
    
    # Calculate metrics only for missing values
    true_missing = all_targets[all_missing_masks]
    pred_missing = all_predictions[all_missing_masks]
    
    # Calculate metrics
    mse = mean_squared_error(true_missing, pred_missing)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_missing, pred_missing)
    
    # Calculate normalized metrics
    mask = true_missing > 0  # Avoid division by zero
    mape = np.mean(np.abs((true_missing[mask] - pred_missing[mask]) / true_missing[mask])) * 100
    
    return {
        "val_loss": avg_loss,
        "val_rec_loss": avg_rec_loss,
        "val_kl_loss": avg_kl_loss,
        "val_imputation_mse": mse,
        "val_imputation_rmse": rmse,
        "val_imputation_mae": mae,
        "val_imputation_mape": mape
    }



def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int, 
    metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Dictionary with metrics
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "metrics": metrics
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")


def plot_losses(train_losses: List[float], val_losses: List[float], save_path: str) -> None:
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()



def evaluate_imputation(
    model: EVChargingDecoder,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str
) -> Dict[str, float]:
    """
    Evaluate model imputation performance.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data containing missing values
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_missing_masks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating imputation"):
            # Move everything to device
            embedding = batch["embedding"].to(device)
            station_id = batch["station_id"].to(device)
            calendar_features = batch["calendar_features"].to(device)
            history = batch["history"].to(device)
            masked_normalized_history = batch['masked_normalized_history'].to(device)
            history_mean = batch['history_mean'].to(device)
            history_std = batch['history_std'].to(device)
            missing_mask = batch["missing_mask"].to(device)
            
            
            # Forward pass - using masked history
            outputs = model.forward(
                llm_embedding=embedding, 
                station_id=station_id, 
                calendar_features=calendar_features, 
                initial_sequence=masked_normalized_history,
                sample=False,
                mask=missing_mask
            )
            
            # Get predictions
            predictions = outputs["mean"]
            predictions = predictions * history_std.unsqueeze(1).unsqueeze(1) + history_mean.unsqueeze(1).unsqueeze(1)
            
            # Append to lists
            all_targets.append(history.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_missing_masks.append(missing_mask.cpu().numpy())
    
    # Concatenate
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_missing_masks = np.concatenate(all_missing_masks, axis=0)
    
    
    os.makedirs(save_dir, exist_ok=True)
        
    # Save data as numpy arrays
    np.save(os.path.join(save_dir, "evaluation_targets.npy"), all_targets)
    np.save(os.path.join(save_dir, "evaluation_predictions.npy"), all_predictions)
    np.save(os.path.join(save_dir, "evaluation_missing_masks.npy"), all_missing_masks)
    
    logger.info(f"Saved evaluation data to {save_dir}")
    
    
    
    # Calculate metrics only for missing values
    true_missing = all_targets[all_missing_masks]
    pred_missing = all_predictions[all_missing_masks]
    
    # Calculate metrics
    mse = mean_squared_error(true_missing, pred_missing)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_missing, pred_missing)
    
    # Calculate normalized metrics
    mask = true_missing > 0  # Avoid division by zero
    mape = np.mean(np.abs((true_missing[mask] - pred_missing[mask]) / true_missing[mask])) * 100
    
    return {
        "imputation_mse": mse,
        "imputation_rmse": rmse,
        "imputation_mae": mae,
        "imputation_mape": mape
    }



def train_imputation_decoder(
    device: torch.device,
    data_points: List[EVChargingDataPoint],
    train_data_points: List[EVChargingDataPoint],
    test_data_points: List[EVChargingDataPoint],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str,
    dataset_name: str,
    load_epoch: int=None,
    load_timestamp: str=None,
    checkpoint_freq: int=20
) -> Tuple[nn.Module, Dict[str, Any], RAGIndex]:
    """
    Train the decoder model for imputation.
    
    Args:
        data_points: List of data points with embeddings
        model_config: Model configuration
        training_config: Training configuration
        output_dir: Directory for saving outputs
        
    Returns:
        Tuple containing:
        - Trained model
        - Dictionary with training history and metrics
        - RAG index with training data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up metrics CSV file
    metrics_csv_path = os.path.join(experiment_dir, "training_metrics.csv")
    setup_metrics_csv(metrics_csv_path)
    
    # Save configs
    with open(os.path.join(experiment_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    with open(os.path.join(experiment_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)
    
    # Set device
    logger.info(f"Using device: {device}")
    
    # Prepare data with RAG index
    rag_index = prep_rag_train_decoder(data_points=train_data_points)
    
    # Create datasets
    train_dataset = EVChargingImputationDataset(
        data_points=train_data_points,
        rag_index=rag_index,
        dataset_name=dataset_name,
        k_neighbors=training_config["k_neighbors"],
        refinement_method=training_config["refinement_method"]
    )
    
    test_dataset = EVChargingImputationDataset(
        data_points=test_data_points,
        rag_index=rag_index,
        dataset_name=dataset_name,
        k_neighbors=training_config["k_neighbors"],
        refinement_method=training_config["refinement_method"]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"]
    )
    
    # Create model
    model = EVChargingDecoder(
        d_emb=model_config["d_emb"],
        d_latent=model_config["d_latent"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        nhead=model_config["nhead"],
        d_station=model_config["d_station"],
        d_calendar=model_config["d_calendar"],
        num_stations=model_config["num_stations"],
        calendar_feature_dim=model_config["calendar_feature_dim"],
        seq_length=model_config["seq_length"],
        output_dim=model_config["output_dim"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )
    
    # Create scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, 
        T_0=training_config["scheduler_t0"], 
        T_mult=training_config["scheduler_t_mult"],
        eta_min=training_config["scheduler_eta_min"]
    )
    
    # Initialize tracking variables
    best_val_rmse = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_rmses = []
    start_epoch = 0
    
    # Update the checkpoint loading section in train_imputation_decoder
    # Load checkpoint if epoch is provided
    if load_epoch is not None:
        # First check if a specific experiment timestamp was provided
        specific_exp_dir = os.path.join(output_dir, f"experiment_{load_timestamp}")
        checkpoint_path = os.path.join(specific_exp_dir, "checkpoints", f"checkpoint_epoch_{load_epoch}.pth")
        logger.info(f"Looking for checkpoint in specified experiment directory: {specific_exp_dir}")
        logger.info(f"Loading checkpoint from epoch {load_epoch}: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        
        # If metrics exist, use them for tracking
        if "metrics" in checkpoint:
            if "val_imputation_rmse" in checkpoint["metrics"]:
                best_val_rmse = checkpoint["metrics"]["val_imputation_rmse"]
                logger.info(f"Loaded best validation RMSE: {best_val_rmse}")
        
        logger.info(f"Resuming training from epoch {start_epoch}")

    
    # Training loop
    for epoch in tqdm(range(start_epoch, training_config["max_epochs"]), desc='Training Decoder'):
        logger.info(f"Epoch {epoch+1}/{training_config['max_epochs']}")
        
        # Calculate KL weight with annealing
        if training_config["use_kl_annealing"]:
            kl_weight = min(
                training_config["max_kl_weight"],
                training_config["max_kl_weight"] * epoch / training_config["kl_annealing_epochs"]
            )
        else:
            kl_weight = training_config["max_kl_weight"]
        
        # Train
        train_metrics = train_epoch(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer, 
            device=device, 
            dataset_name=dataset_name,
            kl_weight=kl_weight,
            sample=training_config["sample_during_training"],
        )
        logger.info(f"Train metrics: {train_metrics}")
        
        # Validate
        val_metrics = validate(
            model=model, 
            dataloader=test_loader, 
            device=device, 
            dataset_name=dataset_name,
            kl_weight=kl_weight,
        )
        logger.info(f"Validation metrics: {val_metrics}")
        
        # Update scheduler
        scheduler.step()
        
        # Track losses
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["val_loss"])
        val_rmses.append(val_metrics["val_imputation_rmse"])
        
        # Save checkpoint - now using RMSE for best model determination
        is_best = val_metrics["val_imputation_rmse"] < best_val_rmse
        save_this_epoch = ((epoch+1) % checkpoint_freq == 0) or is_best
        if is_best:
            best_val_rmse = val_metrics["val_imputation_rmse"]
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update metrics CSV
        update_metrics_csv(
            filename=metrics_csv_path, 
            epoch=epoch, 
            train_metrics=train_metrics, 
            val_metrics=val_metrics, 
            is_best=is_best
        )
        
        if save_this_epoch:
            save_checkpoint(
                model=model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                epoch=epoch, 
                metrics={**train_metrics, **val_metrics},
                checkpoint_dir=checkpoint_dir, 
                is_best=is_best
            )
        
        # Early stopping
        if patience_counter >= training_config["patience"]:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot losses
    plot_losses(
        train_losses=train_losses, 
        val_losses=val_losses, 
        save_path=os.path.join(experiment_dir, "loss_plot.png")
    )
    
    # Load best model
    best_checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pth"), weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    logger.info("Loaded best model for evaluation")
    
    # Evaluate imputation performance
    imputation_metrics = evaluate_imputation(
        model=model, 
        dataloader=test_loader, 
        device=device,
        save_dir=experiment_dir
    )
    logger.info(f"Imputation performance: {imputation_metrics}")
    
    # Save final metrics - update to include best RMSE
    final_metrics = {
        "imputation": imputation_metrics,
        "best_val_loss": min(val_losses),
        "best_val_rmse": best_val_rmse,
        "epochs_trained": len(train_losses)
    }
    
    with open(os.path.join(experiment_dir, "final_metrics.pkl"), "wb") as f:
        pickle.dump(final_metrics, f)
    
    # Return model and training history
    training_history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_rmses": val_rmses,
        "final_metrics": final_metrics
    }
    
    return model, training_history, rag_index


def main(args: dict):
    
    # Example usage
    logger.info("Starting imputation decoder training")
    
    # Define model configuration
    model_config = {
        "d_emb": args['d_emb'],            # LLM embedding dimension
        "d_latent": args['d_latent'],         # Latent space dimension 128
        "d_model": args['d_model'],          # Transformer model dimension 256
        "num_layers": args['num_layers'],         # Number of transformer layers 4
        "nhead": args['nhead'],              # Number of attention heads 8
        "d_station": args['d_station'],         # Station embedding dimension 32
        "d_calendar": args['d_calendar'],        # Calendar embedding dimension 32 
        "num_stations": args['num_stations'],     # Number of unique stations 
        "calendar_feature_dim": args['calendar_feature_dim'],  # Dimension of calendar features 9 
        "seq_length": args['seq_length'],         # Sequence length (7 days) 
        "output_dim": args['output_dim'],         # Output dimension 
        "dropout": args['dropout']           # Dropout probability 0.1  
    }
    
    training_config = {
        "batch_size": args['batch_size'],
        "learning_rate": args['learning_rate'],
        "weight_decay": args['weight_decay'],
        "max_epochs": args['max_epochs'],
        "patience": args['patience'],
        "train_ratio": args['train_ratio'],
        "random_seed": args['random_seed'],
        "num_workers": 1,
        "max_kl_weight": args['max_kl_weight'],
        "use_kl_annealing": args['use_kl_annealing'],
        "kl_annealing_epochs": args['kl_annealing_epochs'],
        "scheduler_t0": args['scheduler_t0'],
        "scheduler_t_mult": args['scheduler_t_mult'],
        "scheduler_eta_min": args['scheduler_eta_min'],
        "missing_ratio": args['missing_ratio'],
        "k_neighbors": args['k_neighbors'],
        "refinement_method": args['refinement_method'],
        "sample_during_training": args['sample_during_training']
    }
    
    city = args['city']
    window_size = args['seq_length']
    poi_radius = args['poi_radius']
    missing_ratio = args['missing_ratio']
    train_data_points = load_saved_data_points(logger=logger, input_path=f'data_{city}/datapoints_ws{window_size}_poiradius{poi_radius}_withembed_maskratio{missing_ratio}_train.pkl')
    test_data_points = load_saved_data_points(logger=logger, input_path=f'data_{city}/datapoints_ws{window_size}_poiradius{poi_radius}_withembed_maskratio{missing_ratio}_test.pkl')
    
    # Train decoder
    model, history, rag_index = train_imputation_decoder(
        device=args['device'] if torch.cuda.is_available() else 'cpu',
        data_points=train_data_points,
        train_data_points=train_data_points,
        test_data_points=test_data_points,
        model_config=model_config,
        training_config=training_config,
        dataset_name=city,
        output_dir=args['output_dir'],
        load_epoch=args['load_epoch'],
        load_timestamp=args['load_timestamp']
    )
    
    logger.info("Imputation decoder training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EV Charging Data Imputation Decoder')
    
    # Device configuration
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to train on (cuda/cpu)')
    
    # Model architecture parameters
    parser.add_argument('--d_emb', default=4096, type=int,
                        help='LLM embedding dimension')
    parser.add_argument('--d_latent', default=256, type=int,
                        help='Latent space dimension')
    parser.add_argument('--d_model', default=256, type=int,
                        help='Transformer model dimension')
    parser.add_argument('--num_layers', default=4, type=int,
                        help='Number of transformer layers')
    parser.add_argument('--nhead', default=8, type=int,
                        help='Number of attention heads')
    parser.add_argument('--d_station', default=32, type=int,
                        help='Station embedding dimension')
    parser.add_argument('--d_calendar', default=32, type=int,
                        help='Calendar embedding dimension')
    parser.add_argument('--calendar_feature_dim', default=9, type=int,
                        help='Dimension of calendar features')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout probability')
    
    # Training configuration
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', default=200, type=int,
                        help='Early stopping patience')
    parser.add_argument('--train_ratio', default=0.8, type=float,
                        help='Ratio of data to use for training vs testing')
    parser.add_argument('--random_seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    # KL annealing parameters
    parser.add_argument('--max_kl_weight', default=0.1, type=float,
                        help='Maximum weight for KL divergence term')
    parser.add_argument('--use_kl_annealing', default=True, type=bool,
                        help='Whether to use KL annealing')
    parser.add_argument('--kl_annealing_epochs', default=10, type=int,
                        help='Number of epochs for KL annealing')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_t0', default=10, type=int,
                        help='Initial restart period for cosine annealing')
    parser.add_argument('--scheduler_t_mult', default=2, type=int,
                        help='Multiplier for restart period')
    parser.add_argument('--scheduler_eta_min', default=1e-6, type=float,
                        help='Minimum learning rate for scheduler')
    
    # Imputation parameters
    parser.add_argument('--sample_during_training', default=True, type=bool,
                        help='Whether to sample from latent distribution during training')
    

    # essential parameters
    parser.add_argument('--city', default='Dundee', type=str, choices=['PaloAlto', 'Boulder', 'Dundee', 'Perth'],
                        help='city of the dataset')
    parser.add_argument('--poi_radius', default=2000, type=int, 
                        help='poi radius')
    parser.add_argument('--seq_length', default=7, type=int,
                        help='Sequence length in days')
    parser.add_argument('--max_epochs', default=3000, type=int,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Training batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--output_dim', default=1, type=int,
                        help='Output dimension (typically 1 for charging demand)')
    parser.add_argument('--missing_ratio', default=0.3, type=float,
                        help='Ratio of values to artificially mark as missing')
    parser.add_argument('--k_neighbors', default=10, type=int,
                        help='Number of neighbors to retrieve from RAG')
    parser.add_argument('--refinement_method', default='attention', type=str,
                        choices=['weighted', 'attention', 'guidance'],
                        help='Method for refining embeddings')
    parser.add_argument('--load_epoch', default=None, type=int,
                    help='Epoch number to load checkpoint from (optional)')
    parser.add_argument('--load_timestamp', default=None, type=str,
                    help='Timestamp of experiment directory to load checkpoint from (e.g., "20250413_120000")')
    
    
    args = parser.parse_args()
    args = vars(args)
    
    city = args['city']
    poi_radius = args['poi_radius']
    seq_length = args['seq_length']
    output_dim = args['output_dim']
    missing_ratio = args['missing_ratio']
    k_neighbors = args['k_neighbors']
    refinement_method = args['refinement_method']
    args['output_dir'] = f'outputs/{city}/poiradius{poi_radius}_seqlen{seq_length}_maskratio{missing_ratio}_kneighbor{k_neighbors}_refinemethod{refinement_method}'
    
    args['num_stations'] = NUM_STATIONS[args['city']]
    
    main(args=args)