import numpy as np
import os
import logging
import json
import argparse

from typing import Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error

from baselines import mean_imputation, zero_imputation, last_observed_imputation, interpolation_imputation
from baselines import knn_imputation, svd_imputation, kalman_imputation, missforest_imputation
from baselines import gan_imputation, autoencoder_imputation
from baselines import lstm_imputation, transformer_imputation, tcn_imputation

from utils import EVChargingDataPoint, load_saved_data_points


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("baselines.log"), logging.StreamHandler()]
)
logger = logging.getLogger("run_baseline.py")


def evaluate_imputation(data_points: List[EVChargingDataPoint]) -> Dict[str, float]:
    """
    Evaluate imputation performance.
    
    Args:
        data_points: List of data points with imputed values
        
    Returns:
        Dictionary with evaluation metrics
    """
    all_targets = []
    all_predictions = []
    all_missing_masks = []
    
    for dp in data_points:
        # Get original history (ground truth)
        history = dp.history
        
        # Get imputed values
        imputed = dp.imputed_history
        
        # Get missing mask
        missing_mask = dp.missing_mask
        
        # Append to lists
        all_targets.append(history)
        all_predictions.append(imputed)
        all_missing_masks.append(missing_mask)
    
    # Concatenate if needed
    if len(all_targets) > 1:
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_missing_masks = np.concatenate(all_missing_masks, axis=0)
    else:
        all_targets = np.array(all_targets[0])
        all_predictions = np.array(all_predictions[0])
        all_missing_masks = np.array(all_missing_masks[0])
    
    # Calculate metrics only for missing values
    true_missing = all_targets[all_missing_masks]
    pred_missing = all_predictions[all_missing_masks]
    
    # Calculate metrics
    mse = mean_squared_error(true_missing, pred_missing)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_missing, pred_missing)
    
    # Calculate normalized metrics
    mask = true_missing > 0  # Avoid division by zero
    mape = np.mean(np.abs((true_missing[mask] - pred_missing[mask]) / true_missing[mask])) * 100 if sum(mask) > 0 else np.nan
    
    return {
        "imputation_mse": mse,
        "imputation_rmse": rmse,
        "imputation_mae": mae,
        "imputation_mape": mape
    }


# Update the run_baselines function to include new methods and train/test flow
def run_baselines(
    train_data_points: List[EVChargingDataPoint],
    test_data_points: List[EVChargingDataPoint],
    baselines: List[str] = ["mean", "zero", "last_observed", "interpolation", "knn", "svd", "kalman", 
                           "missforest", "gan", "autoencoder", "lstm", "transformer", "tcn"],
    output_dir: str = "outputs/baselines",
    knn_neighbors: int=5,
    svd_rank: int=3,
    rf_estimators: int=100,
    epochs: int=100,
    batch_size: int=32,
    hidden_dim: int=64,
    num_layers: int=2,
    nhead: int=4,
    device: str="cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Run specified baseline imputation methods and evaluate them.
    
    Args:
        train_data_points: List of data points for training
        test_data_points: List of data points with missing values for evaluation
        baselines: List of baseline methods to run
        output_dir: Directory to save results
        knn_neighbors: Number of neighbors for KNN
        svd_rank: Rank for SVD approximation
        rf_estimators: Number of estimators for random forest
        epochs: Number of epochs for neural network methods
        batch_size: Batch size for neural network methods
        hidden_dim: Hidden dimension for neural networks
        num_layers: Number of layers for LSTM/Transformer
        nhead: Number of heads for Transformer
        device: Device to use for training (cuda/cpu)
        
    Returns:
        Dictionary with evaluation metrics for each method
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Simple methods that don't need training data
    simple_methods = {
        "mean": mean_imputation,
        "zero": zero_imputation,
        "last_observed": last_observed_imputation,
        "interpolation": interpolation_imputation,
    }
    
    # Statistical methods that can benefit from training data
    stat_methods = {
        "knn": lambda train, test: knn_imputation(train_data=train, test_data=test, n_neighbors=knn_neighbors, output_dir=output_dir),
        "svd": lambda train, test: svd_imputation(train_data=train, test_data=test, rank=svd_rank, output_dir=output_dir),
        "kalman": lambda train, test: kalman_imputation(train_data=train, test_data=test, output_dir=output_dir),
        "missforest": lambda train, test: missforest_imputation(
            train_data=train, test_data=test, logger=logger, n_estimators=rf_estimators, output_dir=output_dir
        ),
    }
    
    # Deep learning methods that require training
    dl_methods = {
        "lstm": lambda train, test: lstm_imputation(
            train_data=train, test_data=test, logger=logger, 
            hidden_dim=hidden_dim, num_layers=num_layers,
            epochs=epochs, batch_size=batch_size, device=device, output_dir=output_dir
        ),
        "transformer": lambda train, test: transformer_imputation(
            train_data=train, test_data=test, logger=logger, 
            d_model=hidden_dim, nhead=nhead, num_layers=num_layers,
            epochs=epochs, batch_size=batch_size, device=device, output_dir=output_dir
        ),
        "tcn": lambda train, test: tcn_imputation(
            train_data=train, test_data=test, logger=logger, 
            num_channels=[hidden_dim] * 3, kernel_size=3, dropout=0.2,
            epochs=epochs, batch_size=batch_size, device=device, output_dir=output_dir
        ),
        "gan": lambda train, test: gan_imputation(
            train_data=train, test_data=test, logger=logger,
            latent_dim=hidden_dim, epochs=epochs, batch_size=batch_size, device=device, output_dir=output_dir
        ),
        "autoencoder": lambda train, test: autoencoder_imputation(
            train_data=train, test_data=test, logger=logger,
            hidden_dim=hidden_dim, epochs=epochs, batch_size=batch_size, device=device, output_dir=output_dir
        ),
    }
    
    # Run selected baselines
    for baseline in baselines:
        logger.info(f"Running {baseline} imputation...")
        
        if baseline in simple_methods:
            # Simple methods that don't need training
            imputed = simple_methods[baseline](test_data_points)
        elif baseline in stat_methods:
            # Statistical methods that can benefit from training data
            imputed = stat_methods[baseline](train_data_points, test_data_points)
        elif baseline in dl_methods:
            # Deep learning methods that require training
            imputed = dl_methods[baseline](train_data_points, test_data_points)
        else:
            logger.warning(f"Unknown baseline method: {baseline}")
            continue
        
        metrics = evaluate_imputation(imputed)
        results[baseline] = metrics
        logger.info(f"{baseline} imputation metrics: {metrics}")
    
    # Save results
    with open(os.path.join(output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main(args):
    # Load both train and test data points
    city = args.city
    window_size = args.window_size
    poi_radius = args.poi_radius
    missing_ratio = args.missing_ratio
    
    # Load training data
    train_data_path = f'data_{city}/datapoints_ws{window_size}_poiradius{poi_radius}_withembed_maskratio{missing_ratio}_train.pkl'
    logger.info(f"Loading training data from {train_data_path}")
    train_data_points = load_saved_data_points(logger=logger, input_path=train_data_path)
    
    # Load test data
    test_data_path = f'data_{city}/datapoints_ws{window_size}_poiradius{poi_radius}_withembed_maskratio{missing_ratio}_test.pkl'
    logger.info(f"Loading test data from {test_data_path}")
    test_data_points = load_saved_data_points(logger=logger, input_path=test_data_path)
    
    # Create output directory    
    output_dir = f'outputs/{args.city}/{args.baseline}_poiradius{args.poi_radius}_seqlen{args.window_size}_maskratio{args.missing_ratio}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which baselines to run
    if args.baseline == "all":
        baselines = ["mean", "zero", "last_observed", "interpolation", "knn", "svd", "kalman", 
                    "missforest", "lstm", "transformer", "tcn", "gan", "autoencoder"]
    else:
        baselines = [args.baseline]
    
    # Run selected baselines
    results = run_baselines(
        train_data_points=train_data_points,
        test_data_points=test_data_points,
        baselines=baselines,
        output_dir=output_dir,
        knn_neighbors=args.knn_neighbors,
        svd_rank=args.svd_rank,
        rf_estimators=args.rf_estimators,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        device=args.device
    )
    
    # Print summary
    logger.info("==== Baseline Imputation Results ====")
    for method, metrics in results.items():
        logger.info(f"{method.upper()} - RMSE: {metrics['imputation_rmse']:.4f}, MAE: {metrics['imputation_mae']:.4f}, MAPE: {metrics['imputation_mape']:.2f}%")
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run baseline imputation methods')
    
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to train on (cuda/cpu)')
    
    parser.add_argument('--city', default='Boulder', type=str, 
                        choices=['PaloAlto', 'Boulder', 'Dundee', 'Perth'],
                        help='City of the dataset')
    parser.add_argument('--window_size', default=7, type=int,
                        help='Window size in days')
    parser.add_argument('--poi_radius', default=2000, type=int,
                        help='POI radius')
    parser.add_argument('--missing_ratio', default=0.1, type=float,
                        help='Ratio of values artificially marked as missing')
    parser.add_argument('--baseline', default='mean', type=str,
                        choices=['all', 'mean', 'zero', 'last_observed', 'interpolation', 
                                'knn', 'svd', 'kalman', 'missforest', 'lstm', 'transformer', 
                                'tcn', 'gan', 'autoencoder'],
                        help='Baseline method to run (or "all" to run all methods)')
    
    
    parser.add_argument('--knn_neighbors', default=5, type=int,
                        help='Number of neighbors for KNN imputation')
    parser.add_argument('--svd_rank', default=3, type=int,
                        help='Rank for SVD matrix factorization')
    parser.add_argument('--rf_estimators', default=100, type=int, 
                        help='Number of estimators for random forest')
    parser.add_argument('--epochs', default=500, type=int,
                        help='Number of epochs for neural network methods')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for neural network methods')
    parser.add_argument('--hidden_dim', default=64, type=int,
                    help='Hidden dimension for neural network models')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='Number of layers for LSTM/Transformer')
    parser.add_argument('--nhead', default=4, type=int,
                        help='Number of attention heads for Transformer')
    
    args = parser.parse_args()
    
    
    
    main(args)