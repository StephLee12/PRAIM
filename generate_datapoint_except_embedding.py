import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import logging
from typing import List

from utils import EVChargingDataPoint, get_surrounding_pois, save_data_points, is_weekend


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("generate_datapoint_except_embedding")




def load_data_points(
    city: str,
    window_size: int,
    poi_radius: int,
    data_dir: str="./data_",
) -> List[EVChargingDataPoint]:
    """
    Load and prepare EV charging data points from multiple cities.
    
    Args:
        city: city name to load data from
        window_size: Size of the sliding window for history
        device: PyTorch device for embedding generation
        data_dir: Base directory for data files
        embedding_batch_size: Batch size for generating embeddings
        
    Returns:
        List of EVChargingDataPoint objects
    """

    
    all_data_points = []
    
    # Initialize LLM embedder
    
    logger.info(f"Processing data for {city}")
    
    # File paths
    daily_data_path = os.path.join(data_dir + city, "daily_data_withNaNs.csv")
    station_info_path = os.path.join(data_dir + city, "daily_data_missing_percentage.csv")
    
    station_lon_lat_path = os.path.join(data_dir + city, "station_lon_lat.pkl")
    
    # Load data
    daily_demand_df = pd.read_csv(daily_data_path, parse_dates=["start_date"], index_col=[0])
    if city != 'PaloAlto': daily_demand_df.set_index('start_date', inplace=True)
    station_start_end_date_df = pd.read_csv(station_info_path, index_col=[0])
    
    with open(station_lon_lat_path, 'rb') as f:
        station_lon_lat_data = pickle.load(f)

    
    # Process each station
    for station_id_num, station_info in tqdm(station_start_end_date_df.iterrows(), desc=f"Processing stations in {city}"):
        station_name = station_info["station_name"]
        
        start_date = pd.to_datetime(station_info["start_date"])
        end_date = pd.to_datetime(station_info["end_date"])
        
        # Extract station data
        station_data = daily_demand_df.copy()
        
        # Get station charging demand data
        station_demand = station_data[station_name].to_frame().reset_index()
        station_demand.columns = ["date", "demand"]
        station_demand['date'] = pd.to_datetime(station_demand['date'])
        station_demand = station_demand[(station_demand['date']>=start_date)&(station_demand['date']<=end_date)].reset_index(drop=True)
        
        # Ensure all dates in the range are included
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        full_df = pd.DataFrame({"date": date_range})
        station_demand = full_df.merge(station_demand, on="date", how="left")
        
        
        # Convert to numpy arrays for easier processing
        dates = station_demand["date"].tolist()
        demands = station_demand["demand"].values.reshape(-1, 1)  # [days, 1]
        missing_mask = np.isnan(demands)
        
        # Fill NaN values with zeros temporarily
        demands = np.nan_to_num(demands, nan=0.0)
        
        
        # Compile station information
        # station_id_num = hash(station_name) % 10000  # Simple hashing for numeric ID
        # Get GIS coordinates if available
        coordinates = station_lon_lat_data[station_name]
        # Get surrounding POIs
        pois = get_surrounding_pois(logger=logger, latitude=coordinates[0], longitude=coordinates[1], radius=poi_radius)
        station_info_dict = {
            'id': station_id_num,
            'name': station_name,
            'location': city,
            'type': 'Public',  # Default, could be enhanced with more metadata
            'coordinates': (coordinates[0], coordinates[1]),
            'surrounding_pois': pois  # Add the POIs here
        }
        
        
        
        # Create sliding windows
        for i in range(len(dates) - window_size + 1):
            window_dates = dates[i:i+window_size]
            window_demands = demands[i:i+window_size]
            window_missing = missing_mask[i:i+window_size]
            
            # # Skip windows where all values are missing
            # if np.all(window_missing):
            #     continue
            
            # Get calendar info for the last date in the window
            last_date = window_dates[-1]
            calendar_info = {
                'year': last_date.year,
                'month': last_date.month,
                'day': last_date.day,
                'day_of_week': last_date.weekday(),
                'is_weekend': is_weekend(last_date),
                # 'is_holiday': False  # Could be improved with a holiday calendar
            }
            
        
            # Create data point without embedding first
            data_point = EVChargingDataPoint(
                station_id=station_id_num,
                history=window_demands,
                missing_mask=window_missing,
                calendar_info=calendar_info,
                station_info=station_info_dict,
                embedding=None  # Will be added later
            )
            
            all_data_points.append(data_point)
    
    
    logger.info(f"Created {len(all_data_points)} data points")
    
    return all_data_points




def process_city_data(
    city: str,
    window_size: int,
    poi_radius: int,
    output_dir: str,
) -> str:
    """
    Process data from multiple cities and save the results.
    
    Args:
        city: List of city names to process
        window_size: Size of the sliding window for history
        batch_size: Batch size for embedding generation
        
    Returns:
        Path to the saved data file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    output_path = os.path.join(output_dir, f"datapoints_ws{window_size}_poiradius{poi_radius}.pkl")
    
    # Load and process data
    data_points = load_data_points(
        city=city,
        window_size=window_size,
        poi_radius=poi_radius,
    )
    
    # Save processed data
    save_data_points(logger=logger, data_points=data_points, output_path=output_path)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate EV charging data points with embeddings")
    parser.add_argument("--city", type=str, default="PaloAlto", help="List of cities to process")
    parser.add_argument("--window_size", type=int, default=7, help="Size of sliding window for history")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding generation")
    parser.add_argument('--poi_radius', default=2000, type=int, help='poi radius')
    
    args = parser.parse_args()
    
    if args.city == 'PaloAlto': output_dir = './data_PaloAlto/'
    elif args.city == 'Boulder': output_dir = './data_Boulder/'
    elif args.city == 'Dundee': output_dir = './data_Dundee/'
    elif args.city == 'Perth': output_dir = './data_Perth/'
        
    
    output_path = process_city_data(
        city=args.city,
        window_size=args.window_size,
        poi_radius=args.poi_radius,
        output_dir=output_dir
    )
    
    print(f"Data processing complete. Saved to: {output_path}")