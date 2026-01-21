import requests
import time
import logging
import pickle
import threading
import pandas as pd 
import numpy as np 
import logging
import pickle
import datetime 


from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import pandas as pd




def agg_hourly_charging_per_station(df: pd.DataFrame) -> pd.DataFrame:
    hourly_charging_lst = []
    for row_idx, row in df.iterrows():
        start_time, end_time, ch_demand, station_name = row['Start Date'], row['End Date'], row['Energy (kWh)'], row['top_station']
        
        start_date = start_time.replace(minute=0, second=0, microsecond=0)
        end_date = end_time.replace(minute=0, second=0, microsecond=0)
        
        if start_date == end_date:
            # Charging session is within one hour
            hourly_charging_lst.append({
                "date": start_date,
                "station_name": station_name,
                "ch_demand": ch_demand
            })
        else:
            # Charging session spans multiple hours
            first_hour_duration = 60 - start_time.minute  # Remaining minutes in the first hour
            first_hour_energy = (first_hour_duration / row["tot_charing_mins"]) * ch_demand

            hourly_charging_lst.append({
                "date": start_date,
                "station_name": station_name,
                "ch_demand": first_hour_energy
            })

            remaining_energy = ch_demand - first_hour_energy
            next_hour = start_date + pd.Timedelta(hours=1)

            while next_hour < end_date:
                # Full hour charging
                hourly_energy = (60 / row["tot_charing_mins"]) * ch_demand
                hourly_charging_lst.append({
                    "date": next_hour,
                    "station_name": station_name,
                    "ch_demand": hourly_energy
                })
                remaining_energy -= hourly_energy
                next_hour += pd.Timedelta(hours=1)

            # Add remaining energy to the last hour
            hourly_charging_lst.append({
                "date": end_date,
                "station_name": station_name,
                "ch_demand": remaining_energy
            })


    # Convert processed data to DataFrame
    agg_ch_df = pd.DataFrame(hourly_charging_lst)

    # Aggregate energy demand by station and hour
    agg_ch_df = agg_ch_df.groupby(["date", "station_name"], as_index=False).sum()
    agg_ch_df['ch_demand'] = agg_ch_df['ch_demand'].apply(lambda x: max(x, 0)) # the lowest charging demand is zero
    
    
    return agg_ch_df



def get_missing_hour_cnt(agg_ch_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Get unique stations
    stations = agg_ch_df["station_name"].unique()
    
    # Get the global min and max date range
    min_date = agg_ch_df["date"].min()
    max_date = agg_ch_df["date"].max()
    
    # Create a complete hourly time range for the entire dataset
    full_time_range = pd.date_range(start=min_date, end=max_date, freq="h")
    
    # Create a full dataframe with all combinations of date and station_name
    full_df = pd.MultiIndex.from_product([full_time_range, stations], names=["date", "station_name"]).to_frame(index=False)
    
    # Merge with the aggregated dataframe
    merged_df = full_df.merge(agg_ch_df, on=["date", "station_name"], how="left")
    
    # Find missing data (rows where ch_demand is NaN)
    missing_df = merged_df[merged_df["ch_demand"].isna()][["date", "station_name"]]
    
    # Get the first and last date for each station
    station_date_ranges = agg_ch_df.groupby("station_name").agg(
        start_date=("date", "min"),
        end_date=("date", "max")
    ).reset_index()
    
    # Initialize missing hours count dataframe with station names
    missing_hours_count_df = pd.DataFrame({"station_name": stations})
    
    # Add station date ranges to the missing hours count dataframe
    missing_hours_count_df = missing_hours_count_df.merge(station_date_ranges, on="station_name", how="left")
    
    # Calculate missing hours count for each station based on their date range
    missing_hours_count_df["missing_hours_count"] = missing_hours_count_df.apply(
        lambda row: len(missing_df[(missing_df["station_name"] == row["station_name"]) & 
                                  (missing_df["date"] >= row["start_date"]) & 
                                  (missing_df["date"] <= row["end_date"])]),
        axis=1
    )
    
    # Calculate total hours in the dataset for each station based on their date range
    missing_hours_count_df["total_hours_per_station"] = missing_hours_count_df.apply(
        lambda row: len(pd.date_range(start=row["start_date"], end=row["end_date"], freq="h")),
        axis=1
    )
    
    # Calculate missing percentage based on station-specific date ranges
    missing_hours_count_df["missing_hours_percentage"] = (
        missing_hours_count_df["missing_hours_count"] / missing_hours_count_df["total_hours_per_station"]
    ) * 100
    
    return merged_df, missing_df, missing_hours_count_df



def get_missing_day_cnt(agg_ch_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Ensure date is in datetime format and extract the daily date
    agg_ch_df = agg_ch_df.copy()
    agg_ch_df["date"] = pd.to_datetime(agg_ch_df["date"])
    date_only_df = agg_ch_df.copy()
    date_only_df["date"] = date_only_df["date"].dt.date

    # Aggregate daily charging demand per station
    daily_ch_df = date_only_df.groupby(["date", "station_name"], as_index=False)["ch_demand"].sum()

    # Get the global min and max date range
    min_date = date_only_df["date"].min()
    max_date = date_only_df["date"].max()

    # Create a complete daily time range for the entire dataset
    full_time_range = pd.date_range(start=min_date, end=max_date, freq="D").date

    # Get unique stations
    stations = daily_ch_df["station_name"].unique()

    # Create a full dataframe with all combinations of date and station_name
    full_daily_df = pd.MultiIndex.from_product([full_time_range, stations], names=["date", "station_name"]).to_frame(index=False)

    # Merge with the aggregated daily dataframe
    merged_daily_df = full_daily_df.merge(daily_ch_df, on=["date", "station_name"], how="left")

    # Find missing daily data (rows where ch_demand is NaN)
    missing_daily_df = merged_daily_df[merged_daily_df["ch_demand"].isna()][["date", "station_name"]]

    # Get the first and last date for each station (use the original datetime format for easier date extraction)
    station_date_ranges = agg_ch_df.groupby("station_name").agg(
        start_date=("date", "min"),
        end_date=("date", "max")
    ).reset_index()
    
    # Convert datetime to date objects for consistency
    station_date_ranges["start_date"] = station_date_ranges["start_date"].dt.date
    station_date_ranges["end_date"] = station_date_ranges["end_date"].dt.date

    # Initialize missing days count dataframe with station names
    missing_days_count = pd.DataFrame({"station_name": stations})
    
    # Add station date ranges to the missing days count dataframe
    missing_days_count = missing_days_count.merge(station_date_ranges, on="station_name", how="left")
    
    # Calculate missing days count for each station based on their date range
    missing_days_count["missing_days_count"] = missing_days_count.apply(
        lambda row: len(missing_daily_df[(missing_daily_df["station_name"] == row["station_name"]) & 
                                       (missing_daily_df["date"] >= row["start_date"]) & 
                                       (missing_daily_df["date"] <= row["end_date"])]),
        axis=1
    )
    
    # Calculate total days in the dataset for each station based on their date range
    missing_days_count["total_days_per_station"] = missing_days_count.apply(
        lambda row: (row["end_date"] - row["start_date"]).days + 1,
        axis=1
    )
    
    # Calculate missing percentage based on station-specific date ranges
    missing_days_count["missing_days_percentage"] = (
        missing_days_count["missing_days_count"] / missing_days_count["total_days_per_station"]
    ) * 100
    
    return merged_daily_df, missing_daily_df, missing_days_count





@dataclass
class EVChargingDataPoint:
    """Data structure for EV charging data point."""
    station_id: int
    history: np.ndarray  # Shape [seq_length, features]
    missing_mask: np.ndarray  # Boolean mask, True where values are missing
    calendar_info: Dict[str, Any]  # Year, month, day, weekday, etc.
    station_info: Dict[str, Any]  # GIS and station metadata
    embedding: np.ndarray  # LLM-generated embedding
    
    
    
    
def get_surrounding_pois(logger: logging.Logger, latitude: float, longitude: float, radius: int=500):
    """
    Get POIs surrounding a location using direct Overpass API.
    
    Args:
        logger: Logger instance for logging
        latitude: Location latitude
        longitude: Location longitude
        radius: Search radius in meters
        
    Returns:
        List of POI names and types
    """
    logger.info(f"Getting POIs for coordinates: {latitude}, {longitude}")
    
    # Add a POI cache to avoid duplicate queries
    poi_cache = {}
    poi_cache_lock = threading.Lock()
    POI_CACHE_PATH = "poi_cache.pkl"

    # Try to load existing POI cache
    try:
        with open(POI_CACHE_PATH, 'rb') as f:
            poi_cache = pickle.load(f)
    except FileNotFoundError:
        poi_cache = {}

    
    # Create cache key
    cache_key = f"{latitude:.5f}_{longitude:.5f}_{radius}"
    
    # Check cache first
    with poi_cache_lock:
        if cache_key in poi_cache:
            logger.info(f"Found POIs in cache for {latitude}, {longitude}")
            return poi_cache[cache_key]
    
    # Define a simpler and more reliable query
    # First, focusing only on the most important amenities to avoid overloading
    overpass_url = "https://overpass-api.de/api/interpreter"

    # Enrich query with more relevant POIs for EV charging patterns
    overpass_query = f"""
    [out:json][timeout:90];
    (
    /* Food & Drink - places where people might charge while eating */
    node["amenity"~"restaurant|cafe|fast_food|bar|pub"](around:{radius},{latitude},{longitude});
    way["amenity"~"restaurant|cafe|fast_food|bar|pub"](around:{radius},{latitude},{longitude});
    
    /* Education & Work - common charging locations during day */
    node["amenity"~"school|university|college|library"](around:{radius},{latitude},{longitude});
    way["amenity"~"school|university|college|library"](around:{radius},{latitude},{longitude});
    node["office"](around:{radius},{latitude},{longitude});
    way["office"](around:{radius},{latitude},{longitude});
    
    /* Transportation hubs - multimodal connection points */
    node["public_transport"="station"](around:{radius},{latitude},{longitude});
    way["public_transport"="station"](around:{radius},{latitude},{longitude});
    node["amenity"="parking"](around:{radius},{latitude},{longitude});
    way["amenity"="parking"](around:{radius},{latitude},{longitude});
    node["amenity"~"bus_station|taxi"](around:{radius},{latitude},{longitude});
    
    /* Shopping - common charging locations */
    node["shop"~"supermarket|mall|department_store|convenience"](around:{radius},{latitude},{longitude});
    way["shop"~"supermarket|mall|department_store"](around:{radius},{latitude},{longitude});
    
    /* Medical facilities - longer duration stays */
    node["amenity"~"hospital|clinic|doctors"](around:{radius},{latitude},{longitude});
    way["amenity"~"hospital|clinic"](around:{radius},{latitude},{longitude});
    
    /* Leisure - entertainment venues */
    node["leisure"~"fitness_centre|sports_centre|cinema"](around:{radius},{latitude},{longitude});
    way["leisure"~"fitness_centre|sports_centre|cinema"](around:{radius},{latitude},{longitude});
    
    /* Hotels - tourist/visitor charging */
    node["tourism"="hotel"](around:{radius},{latitude},{longitude});
    way["tourism"="hotel"](around:{radius},{latitude},{longitude});
    
    /* Other EV charging stations - clustering effect */
    node["amenity"="charging_station"](around:{radius},{latitude},{longitude});
    );
    out body;
    """ 

    
    logger.info("Sending simplified query to Overpass API...")
    
    try:
        # Execute query with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(overpass_url, params={'data': overpass_query})
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Query successful with {len(data.get('elements', []))} elements")
                    break
                else:
                    logger.warning(f"Request failed with status code {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        logger.warning(f"Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    return []
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error: {e}. Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                logger.error(f"Failed after {max_retries} attempts: {e}")
                return []
        
        # Extract POI information
        pois = []
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            if not tags:
                continue
                
            name = tags.get('name', '')
            poi_type = None
            
            # Try to determine POI type
            if 'amenity' in tags:
                poi_type = tags['amenity']
            elif 'shop' in tags:
                poi_type = tags['shop']
            elif 'building' in tags:
                poi_type = tags['building']
                
            # Skip entries with no identifiable type
            if not poi_type:
                continue
                
            # Use name if available, otherwise use type
            poi_name = name if name else f"{poi_type.replace('_', ' ').capitalize()}"
            
            if poi_name:
                poi_info = f"{poi_name} ({poi_type})"
                if poi_info not in pois:  # Avoid duplicates
                    pois.append(poi_info)
        
        # Limit results
        pois = pois[:50]
        
        logger.info(f"Found {len(pois)} POIs near coordinates {latitude}, {longitude}")
        
        # Save to cache
        with poi_cache_lock:
            poi_cache[cache_key] = pois
            # Save cache every 10 new entries
            if len(poi_cache) % 10 == 0:
                with open(POI_CACHE_PATH, 'wb') as f:
                    pickle.dump(poi_cache, f)
        
        return pois
        
    except Exception as e:
        logger.error(f"Error fetching POIs: {e}")
        return []
    
    

def save_data_points(logger: logging.Logger, data_points: List[EVChargingDataPoint], output_path: str) -> None:
    """Save data points to a pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(data_points, f)
    logger.info(f"Saved {len(data_points)} data points to {output_path}")


def load_saved_data_points(logger: logging.Logger, input_path: str) -> List[EVChargingDataPoint]:
    """Load data points from a pickle file."""
    with open(input_path, 'rb') as f:
        data_points = pickle.load(f)
    logger.info(f"Loaded {len(data_points)} data points from {input_path}")
    
    return data_points


def is_weekend(date: datetime.datetime) -> bool:
    """Check if date is weekend (Saturday=5, Sunday=6)"""
    return date.weekday() >= 5



def get_dp_demand_mean_std(dps: EVChargingDataPoint) -> list[float, float]:
    demand_lst = []
    for dp in dps:
        demand_lst.append(float(dp.history[-1][0]))
    
    return np.mean(demand_lst), np.std(demand_lst)