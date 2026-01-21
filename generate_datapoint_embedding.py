import argparse
import os
from tqdm import tqdm
import logging
from typing import List
import torch

from utils import EVChargingDataPoint, load_saved_data_points, save_data_points
from llm_embedder import LLMEmbedder




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("generate_datapoint_embedding")



def generate_embedding(
    city: str,
    device: torch.device,
    window_size: int,
    poi_radius: int,
    output_dir: str,
    embedding_batch_size: int=1
) -> List[EVChargingDataPoint]:
    """
    Load and prepare EV charging data points from multiple cities.
    
    Args:
        city: city name to load data from
        window_size: Size of the sliding window for history
        device: PyTorch device for embedding generation
        embedding_batch_size: Batch size for generating embeddings
        
    Returns:
        List of EVChargingDataPoint objects
    """


    # Initialize LLM embedder
    embedder = LLMEmbedder(device=device)
        
    logger.info(f"Processing data for {city}")
    
    path = os.path.join(output_dir, f"datapoints_ws{window_size}_poiradius{poi_radius}.pkl")
    all_data_points = load_saved_data_points(logger=logger, input_path=path)

    # Generate embeddings in batches
    logger.info(f"Generating embeddings for {len(all_data_points)} data points")
    embeddings = []
    for i in tqdm(range(0, len(all_data_points), embedding_batch_size), desc="Generating embeddings"):
        batch = all_data_points[i:i+embedding_batch_size]
        batch_embeddings = []
        
        for dp in batch:
            # Generate embedding
            emb = embedder.generate_embedding(dp)
            batch_embeddings.append(emb)
        
        embeddings.extend(batch_embeddings)
    
    # Add embeddings to data points
    for i, dp in enumerate(all_data_points):
        dp.embedding = embeddings[i]
    
    logger.info(f"Created {len(all_data_points)} data points")
    
    new_path = os.path.join(output_dir, f"datapoints_ws{window_size}_poiradius{poi_radius}_withembed.pkl")
    save_data_points(logger=logger, data_points=all_data_points, output_path=new_path)
    
    return new_path



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Generate EV charging data points with embeddings")
    parser.add_argument("--city", type=str, default="PaloAlto", help="List of cities to process")
    parser.add_argument("--window_size", type=int, default=7, help="Size of sliding window for history")
    parser.add_argument('--poi_radius', default=2000, type=int, help='poi radius')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    
    args = parser.parse_args()
    
    if args.city == 'PaloAlto': output_dir = './data_PaloAlto/'
    elif args.city == 'Boulder': output_dir = './data_Boulder/'
    elif args.city == 'Dundee': output_dir = './data_Dundee/'
    elif args.city == 'Perth': output_dir = './data_Perth/'
        
    
    output_path = generate_embedding(
        city=args.city,
        device=args.device,
        window_size=args.window_size,
        poi_radius=args.poi_radius,
        output_dir=output_dir
    )
    
    print(f"Data processing complete. Saved to: {output_path}")