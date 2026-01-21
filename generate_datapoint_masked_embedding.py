import argparse
import torch 
import numpy as np
import logging
from typing import List
import random
from tqdm import tqdm
import os 

from utils import EVChargingDataPoint, load_saved_data_points, save_data_points
from llm_embedder import LLMEmbedder



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger("generate_datapoint_masked_embedding.py")




def create_artificial_missing_data(
    data_points: List[EVChargingDataPoint], 
    missing_ratio: float=0.2,
    random_seed: int=42
) -> List[EVChargingDataPoint]:
    """
    Create copies of data points with artificial missing values.
    
    Args:
        data_points: List of complete data points
        missing_ratio: Ratio of values to mark as missing
        random_seed: Random seed for reproducibility
        
    Returns:
        List of data points with artificial missing values
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    result = []
    for dp in data_points:
        # Skip points that already have missing values
        if np.any(dp.missing_mask):
            continue
            
        # Create a copy
        new_dp = EVChargingDataPoint(
            station_id=dp.station_id,
            history=dp.history.copy(),
            missing_mask=np.zeros_like(dp.missing_mask),
            calendar_info=dp.calendar_info.copy(),
            station_info=dp.station_info.copy(),
            embedding=None
        )
        
        # Randomly select indices to mark as missing
        seq_length = len(dp.history)
        num_missing = int(seq_length * missing_ratio)
        missing_indices = random.sample(range(seq_length), num_missing)
        
        # Mark as missing
        new_dp.missing_mask[missing_indices] = True
        
        # Add to result
        result.append(new_dp)
    
    logger.info(f"Created {len(result)} data points with artificial missing values")
    
    return result


def generate_masked_embedding(
    missing_ratio: float,
    city: str, 
    device: torch.device,
    window_size: int,
    poi_radius: int,
    output_dir: str,
    train_ratio: float=0.8,
    random_seed: int=42,
    embedding_batch_size: int=1
) -> tuple[str, str]:
    """
    Prepare training, validation, and test data with RAG index.
    
    Args:
        data_points: List of complete data points
        train_ratio: Ratio of data for training
        missing_ratio: Ratio of values to mark as missing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - RAG index with complete training data
        - Training data points with artificial missing values
        - Validation data points with artificial missing values
        - Test data points with artificial missing values
    """
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    data_points = load_saved_data_points(logger=logger, input_path=f'data_{city}/datapoints_ws{window_size}_poiradius{poi_radius}_withembed.pkl')
    
    # Filter data points to only include those with embeddings and no missing values
    complete_data = [dp for dp in data_points if not np.any(dp.missing_mask)]
    
    if len(complete_data) == 0:
        raise ValueError("No complete data points with embeddings found")
    
    # Split into train, validation, test
    random.shuffle(complete_data)
    train_size = int(len(complete_data) * train_ratio)
    
    train_data = complete_data[:train_size]
    test_data = complete_data[train_size:]
    
    logger.info(f"Data split: Train={len(train_data)}, Test={len(test_data)}")
    

    # Create artificial missing data for each set
    train_missing = create_artificial_missing_data(
        data_points=train_data, 
        missing_ratio=missing_ratio,
        random_seed=random_seed
    )
    
    test_missing = create_artificial_missing_data(
        data_points=test_data, 
        missing_ratio=missing_ratio,
        random_seed=random_seed+2
    )
    
    
    embedder = LLMEmbedder(device=device)
    logger.info(f"Processing data for {city}")
    train_embeddings = []
    for i in tqdm(range(0, len(train_missing), embedding_batch_size), desc="Generating embeddings"):
    # for i in tqdm(range(0, 2, embedding_batch_size), desc="Generating embeddings"):
        batch = train_missing[i:i+embedding_batch_size]
        batch_embeddings = []
        
        for dp in batch:
            # Generate embedding
            emb = embedder.generate_embedding(dp)
            batch_embeddings.append(emb)
        
        train_embeddings.extend(batch_embeddings)
    for i, dp in enumerate(train_missing):
    # for i, dp in enumerate(train_missing[:2]):
        dp.embedding = train_embeddings[i]
    
    test_embeddings = []
    for i in tqdm(range(0, len(test_missing), embedding_batch_size), desc="Generating embeddings"):
    # for i in tqdm(range(0, 2, embedding_batch_size), desc="Generating embeddings"):
        batch = test_missing[i:i+embedding_batch_size]
        batch_embeddings = []
        
        for dp in batch:
            # Generate embedding
            emb = embedder.generate_embedding(dp)
            batch_embeddings.append(emb)
        
        test_embeddings.extend(batch_embeddings)
    for i, dp in enumerate(test_missing):
    # for i, dp in enumerate(test_missing[:2]):
        dp.embedding = test_embeddings[i]
    
    
    
    logger.info(f"Created {len(data_points)} data points")
    
    train_path = os.path.join(output_dir, f"datapoints_ws{window_size}_poiradius{poi_radius}_withembed_maskratio{missing_ratio}_train.pkl")
    test_path = os.path.join(output_dir, f"datapoints_ws{window_size}_poiradius{poi_radius}_withembed_maskratio{missing_ratio}_test.pkl")

    save_data_points(logger=logger, data_points=train_missing, output_path=train_path)
    save_data_points(logger=logger, data_points=test_missing, output_path=test_path)
    
    
    return train_path, test_path
    

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Generate EV charging data points with embeddings")
    parser.add_argument('--missing_ratio', type=float, default=.3, help='missing ratio')
    parser.add_argument("--city", type=str, default="PaloAlto", help="List of cities to process")
    parser.add_argument("--window_size", type=int, default=7, help="Size of sliding window for history")
    parser.add_argument('--poi_radius', default=2000, type=int, help='poi radius')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    
    args = parser.parse_args()
    
    if args.city == 'PaloAlto': output_dir = './data_PaloAlto/'
    elif args.city == 'Boulder': output_dir = './data_Boulder/'
    elif args.city == 'Dundee': output_dir = './data_Dundee/'
    elif args.city == 'Perth': output_dir = './data_Perth/'
        
    
    train_path, test_path = generate_masked_embedding(
        missing_ratio=args.missing_ratio,
        city=args.city,
        device=args.device,
        window_size=args.window_size,
        poi_radius=args.poi_radius,
        output_dir=output_dir
    )
    
    print(f"Data processing complete. Saved to: {train_path}")
    