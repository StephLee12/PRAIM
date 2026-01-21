import torch
import torch.nn as nn 
import os
import random 
import numpy as np
import pandas as pd
import logging 
import joblib

from copy import deepcopy
from typing import List
from tqdm import tqdm

# Add these imports at the top of the file
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.experimental import enable_iterative_imputer  # Need this to use IterativeImputer
from sklearn.impute import IterativeImputer
from pykalman import KalmanFilter


from utils import EVChargingDataPoint


def mean_imputation(data_points: List[EVChargingDataPoint]) -> List[EVChargingDataPoint]:
    """
    Mean imputation - replace missing values with mean of observed values.
    
    Args:
        data_points: List of data points with missing values
        
    Returns:
        List of imputed data points
    """
    imputed_data = []
    
    for dp in tqdm(data_points, desc="Mean imputation"):
        # Get the original and masked histories
        history = dp.history.copy()  # Original data with true values
        missing_mask = dp.missing_mask.copy()  # Boolean mask for missing values
        
        # Calculate mean of observed values for each day (across time dimension)
        observed_values = history[~missing_mask]
        if len(observed_values) > 0:
            mean_value = np.mean(observed_values)
        else:
            mean_value = 0.0  # Default if all values are missing
            
        # Replace missing values with mean
        imputed = history.copy()
        imputed[missing_mask] = mean_value
        
        # Store the result
        dp_imputed = deepcopy(dp)
        dp_imputed.imputed_history = imputed
        imputed_data.append(dp_imputed)
        
    return imputed_data



def zero_imputation(data_points: List[EVChargingDataPoint]) -> List[EVChargingDataPoint]:
    """
    Zero imputation - replace missing values with zeros.
    
    Args:
        data_points: List of data points with missing values
        
    Returns:
        List of imputed data points
    """
    imputed_data = []
    
    for dp in tqdm(data_points, desc="Zero imputation"):
        # Get the masked history (already has zeros at missing positions)
        history = dp.history.copy()
        missing_mask = dp.missing_mask.copy()
        
        imputed = history.copy()
        imputed[missing_mask] = 0.0
        
        # Store the result (masked history already has zeros at missing positions)
        dp_imputed = deepcopy(dp)
        dp_imputed.imputed_history = imputed
        imputed_data.append(dp_imputed)
        
    return imputed_data



def last_observed_imputation(data_points: List[EVChargingDataPoint]) -> List[EVChargingDataPoint]:
    """
    Last-observed imputation - fill missing values with the last observed value.
    
    Args:
        data_points: List of data points with missing values
        
    Returns:
        List of imputed data points
    """
    imputed_data = []
    
    for dp in tqdm(data_points, desc="Last-observed imputation"):
        # Get the original and masked histories
        history = dp.history.copy()  # Original data with true values
        missing_mask = dp.missing_mask.copy()  # Boolean mask for missing values
        
        # Create a copy for imputation
        imputed = history.copy()
        
        # Apply last-observed-carried-forward logic
        seq_length = history.shape[0]
        
        for i in range(seq_length):
            if i > 0 and missing_mask[i]:  # If current value is missing and not first element
                imputed[i] = imputed[i-1]  # Use the previous value
                
        # Store the result
        dp_imputed = deepcopy(dp)
        dp_imputed.imputed_history = imputed
        imputed_data.append(dp_imputed)
        
    return imputed_data




def interpolation_imputation(data_points: List[EVChargingDataPoint]) -> List[EVChargingDataPoint]:
    """
    Linear interpolation - fill missing values using linear interpolation.
    
    Args:
        data_points: List of data points with missing values
        
    Returns:
        List of imputed data points
    """
    imputed_data = []
    
    for dp in tqdm(data_points, desc="Interpolation imputation"):
        # Get the original and masked histories
        history = dp.history.copy()  # Original data with true values
        missing_mask = dp.missing_mask.copy()  # Boolean mask for missing values
        
        # Create a pandas Series for easy interpolation
        # Convert to a DataFrame where columns are the features/output dimensions
        seq_length = history.shape[0]
        output_dim = history.shape[1] if len(history.shape) > 1 else 1
        
        # Reshape for pandas if needed
        if output_dim == 1:
            series_data = pd.Series(history.flatten())
            # Create a mask where True indicates non-missing values
            valid_mask = ~missing_mask.flatten()
            
            # Use pandas interpolation - only works on non-missing values
            series_data[~valid_mask] = np.nan  # Set missing values to NaN
            interpolated = series_data.interpolate(method='linear', limit_direction='both').values
            
            # Reshape back if needed
            imputed = interpolated.reshape(seq_length, 1)
        else:
            # For multi-dimensional data
            df = pd.DataFrame(history)
            for col in df.columns:
                # Mark missing values as NaN
                df.loc[missing_mask[:, col], col] = np.nan
                # Interpolate
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            imputed = df.values
                
        # Fill any remaining NaN values with forward/backward fill or zeros
        if np.isnan(imputed).any():
            # Try forward fill
            if output_dim == 1:
                # series = pd.Series(imputed)
                series = pd.DataFrame(imputed)
                imputed = series.fillna(method='ffill').fillna(method='bfill').fillna(0).values
            else:
                df = pd.DataFrame(imputed)
                imputed = df.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        # Store the result
        dp_imputed = deepcopy(dp)
        dp_imputed.imputed_history = imputed
        imputed_data.append(dp_imputed)
        
    return imputed_data




def save_ml_model(model, model_type: str, station_id: int, output_dir: str):
    """
    Save a trained machine learning model to disk.
    
    Args:
        model: The trained model to save
        model_type: Type of model (knn, svd, kalman, missforest)
        station_id: The station ID for this model
        output_dir: Directory to save model
    """
    # Create subdirectory for model type
    model_dir = os.path.join(output_dir, f"{model_type}_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create filename with station_id
    filename = f"station_{station_id}_{model_type}.pkl"
    save_path = os.path.join(model_dir, filename)
    
    # Save model using joblib (better for sklearn models)
    joblib.dump(model, save_path)
    
    return save_path



def knn_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint], 
                   output_dir: str, n_neighbors: int=5) -> List[EVChargingDataPoint]:
    """
    KNN imputation - train on training data and impute test data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points for imputation
        n_neighbors: Number of neighbors for KNN
        
    Returns:
        List of imputed test data points
    """
    imputed_test_data = []
    
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="KNN imputation"):
        # Get training data for this station (if available)
        train_dps = train_stations_data[station_id]
        
        # If no training data for this station, use the test data itself
        if len(train_dps) < 3:  # Need at least a few samples
            # Extract features for KNN from test data
            X_test = np.array([dp.history.flatten() for dp in test_dps])
            test_missing_masks = np.array([dp.missing_mask.flatten() for dp in test_dps])
            
            # Replace masked values with NaN for KNNImputer
            for i in range(len(X_test)):
                X_test[i][test_missing_masks[i]] = np.nan
            
            # Fit and transform using KNNImputer on test data only
            imputer = KNNImputer(n_neighbors=min(n_neighbors, len(X_test)), weights="distance")
            X_imputed = imputer.fit_transform(X_test)
        else:
            # Extract features from training data for fitting
            X_train = np.array([dp.history.flatten() for dp in train_dps])  # Use complete history for training
            
            # Extract features from test data for transformation
            X_test = np.array([dp.history.flatten() for dp in test_dps])
            test_missing_masks = np.array([dp.missing_mask.flatten() for dp in test_dps])
            
            # Replace masked values with NaN for KNNImputer
            for i in range(len(X_test)):
                X_test[i][test_missing_masks[i]] = np.nan
            
            # Fit on training data, transform test data
            imputer = KNNImputer(n_neighbors=min(n_neighbors, len(X_train)), weights="distance")
            imputer.fit(X_train)
            X_imputed = imputer.transform(X_test)
            
            # Save the trained imputer
            save_path = save_ml_model(imputer, "knn", station_id, output_dir)
            print(f"Saved KNN model for station {station_id} to {save_path}")
            
        
        # Put imputed values back into test data points
        for i, dp in enumerate(test_dps):
            dp_imputed = deepcopy(dp)
            imputed = dp.history.copy()
            imputed[dp.missing_mask] = X_imputed[i][dp.missing_mask.flatten()].reshape(dp.missing_mask.sum())
            dp_imputed.imputed_history = imputed
            imputed_test_data.append(dp_imputed)
    
    return imputed_test_data



def svd_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint], 
                  output_dir: str, rank: int=3, max_iter: int=10) -> List[EVChargingDataPoint]:
    """
    SVD/Matrix Factorization imputation - train on training data and impute test data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points for imputation
        rank: Rank for SVD approximation
        max_iter: Maximum number of iterations
        
    Returns:
        List of imputed test data points
    """
    imputed_test_data = []
    
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="SVD imputation"):
        # Get training data for this station (if available)
        train_dps = train_stations_data[station_id]
        
        # Check if we have enough data for SVD
        if len(train_dps) <= 3 or len(test_dps) <= 1:
            # Fall back to mean imputation if not enough data
            for dp in test_dps:
                dp_imputed = deepcopy(dp)
                imputed = dp.history.copy()
                observed_values = dp.history[~dp.missing_mask]
                mean_val = np.mean(observed_values) if len(observed_values) > 0 else 0.0
                imputed[dp.missing_mask] = mean_val
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
            continue
                
        # Get data dimensions
        seq_length = test_dps[0].history.shape[0]
        
        # Create matrices for training data
        if len(train_dps) > 0:
            train_matrix = np.array([dp.history.flatten() for dp in train_dps])
            
            # Create matrices for test data
            test_matrix = np.zeros((len(test_dps), seq_length))
            test_mask_matrix = np.zeros((len(test_dps), seq_length), dtype=bool)
            
            for i, dp in enumerate(test_dps):
                test_matrix[i, :] = dp.history.flatten()
                test_mask_matrix[i, :] = dp.missing_mask.flatten()
                
            # Mark missing values as NaN
            test_matrix_with_nan = test_matrix.copy()
            test_matrix_with_nan[test_mask_matrix] = np.nan
            
            # Use IterativeImputer with SVD estimator for matrix completion
            svd_rank = min(rank, min(test_matrix.shape)-1, min(train_matrix.shape)-1)
            if svd_rank < 2:  # SVD needs at least 2 components
                # Fall back to mean imputation if not enough data
                mean_val = np.mean(train_matrix)
                for i, dp in enumerate(test_dps):
                    dp_imputed = deepcopy(dp)
                    imputed = dp.history.copy()
                    imputed[dp.missing_mask] = mean_val
                    dp_imputed.imputed_history = imputed
                    imputed_test_data.append(dp_imputed)
            else:
                # Create and fit the model
                # estimator = TruncatedSVD(n_components=svd_rank)
                pipeline = make_pipeline(TruncatedSVD(n_components=svd_rank), LinearRegression())
                imputer = IterativeImputer(
                    estimator=pipeline,
                    max_iter=max_iter,
                    random_state=42,
                    verbose=0
                )
                imputer.fit(train_matrix)  # Fit on training data
                imputed_matrix = imputer.transform(test_matrix_with_nan)  # Transform test data
                
                
                
                # Save the trained imputer
                save_path = save_ml_model(imputer, "svd", station_id, output_dir)
                print(f"Saved SVD model for station {station_id} to {save_path}")
                
                
                
                
                # Put imputed values back into test data points
                for i, dp in enumerate(test_dps):
                    dp_imputed = deepcopy(dp)
                    imputed = dp.history.copy()
                    flat_mask = dp.missing_mask.flatten()
                    imputed[dp.missing_mask] = imputed_matrix[i][flat_mask]
                    dp_imputed.imputed_history = imputed
                    imputed_test_data.append(dp_imputed)
        else:
            # No training data, fall back to simpler method
            for dp in test_dps:
                dp_imputed = deepcopy(dp)
                imputed = interpolation_imputation([dp])[0].imputed_history
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
    
    return imputed_test_data




def kalman_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint],
                     output_dir: str) -> List[EVChargingDataPoint]:
    """
    Kalman Filter imputation - estimate parameters from training data and impute test data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points for imputation
        output_dir: Directory to save trained models
        
    Returns:
        List of imputed test data points
    """
    imputed_test_data = []
    
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="Kalman Filter imputation"):
        # Get training data for this station (if available)
        train_dps = train_stations_data[station_id]
        
        # Store Kalman Filter parameters for this station
        kalman_params = {}
        
        # Process each test data point
        for dp in test_dps:
            # Get the masked history and missing mask
            masked_history = dp.history.copy()
            missing_mask = dp.missing_mask.copy()
            
            # Create a copy for imputation
            imputed = masked_history.copy()
            
            # Check if multi-dimensional or single-dimensional
            is_multidimensional = len(masked_history.shape) > 1 and masked_history.shape[1] > 1
            
            if is_multidimensional:
                # Handle each dimension separately
                for dim in range(masked_history.shape[1]):
                    data = masked_history[:, dim]
                    mask = missing_mask[:, dim]
                    
                    # Mark missing values as NaN for Kalman Filter
                    data_with_nan = data.copy()
                    data_with_nan[mask] = np.nan 
                    
                    # Initialize Kalman Filter with parameters from training data
                    if len(train_dps) > 0:
                        # Extract same dimension from training data
                        train_data_dim = np.array([dp.history[:, dim] for dp in train_dps])
                        
                        # Estimate parameters from training data
                        transition_cov = np.var(np.diff(train_data_dim, axis=1), axis=(0, 1))
                        observation_cov = np.var(train_data_dim, axis=(0, 1))
                        
                        # Ensure positive covariance
                        transition_cov = max(0.01, transition_cov)
                        observation_cov = max(1.0, observation_cov)
                        
                        initial_mean = np.nanmean(data_with_nan)
                        if np.isnan(initial_mean):
                            initial_mean = np.mean(train_data_dim)
                            
                        # Store parameters for this dimension
                        kalman_params[f'dim_{dim}'] = {
                            'transition_cov': transition_cov,
                            'observation_cov': observation_cov,
                            'initial_mean': initial_mean
                        }
                    else:
                        # Default parameters if no training data
                        transition_cov = 0.01
                        observation_cov = 1.0
                        initial_mean = np.nanmean(data_with_nan)
                        if np.isnan(initial_mean):
                            initial_mean = 0.0
                            
                        kalman_params[f'dim_{dim}'] = {
                            'transition_cov': transition_cov,
                            'observation_cov': observation_cov,
                            'initial_mean': initial_mean
                        }
                    
                    kf = KalmanFilter(
                        initial_state_mean=initial_mean,
                        n_dim_obs=1,
                        transition_matrices=np.array([[1.0]]),
                        observation_matrices=np.array([[1.0]]),
                        transition_covariance=np.array([[transition_cov]]),
                        observation_covariance=np.array([[observation_cov]]),
                        initial_state_covariance=np.array([[1.0]])
                    )
                    
                    # Apply Kalman Filter
                    smoothed_state_means, _ = kf.smooth(data_with_nan.reshape(-1, 1))
                    
                    # Update imputed data for this dimension
                    imputed[:, dim][mask] = smoothed_state_means.flatten()[mask]
            else:
                # Single-dimensional case
                data = masked_history
                
                # Mark missing values as NaN for Kalman Filter
                data_with_nan = data.copy()
                data_with_nan[missing_mask] = np.nan 
                
                # Initialize Kalman Filter with parameters from training data
                if len(train_dps) > 0:
                    # Extract data from training data
                    train_data_flat = np.array([dp.history.flatten() for dp in train_dps])
                    
                    # Estimate parameters from training data
                    transition_cov = np.var(np.diff(train_data_flat, axis=1), axis=(0, 1))
                    observation_cov = np.var(train_data_flat, axis=(0, 1))
                    
                    # Ensure positive covariance
                    transition_cov = max(0.01, transition_cov)
                    observation_cov = max(1.0, observation_cov)
                    
                    initial_mean = np.nanmean(data_with_nan)
                    if np.isnan(initial_mean):
                        initial_mean = np.mean(train_data_flat)
                        
                    kalman_params['single_dim'] = {
                        'transition_cov': transition_cov,
                        'observation_cov': observation_cov,
                        'initial_mean': initial_mean
                    }
                else:
                    # Default parameters if no training data
                    transition_cov = 0.01
                    observation_cov = 1.0
                    initial_mean = np.nanmean(data_with_nan)
                    if np.isnan(initial_mean):
                        initial_mean = 0.0
                        
                    kalman_params['single_dim'] = {
                        'transition_cov': transition_cov,
                        'observation_cov': observation_cov,
                        'initial_mean': initial_mean
                    }
                
                kf = KalmanFilter(
                    initial_state_mean=initial_mean,
                    n_dim_obs=1,
                    transition_matrices=np.array([[1.0]]),
                    observation_matrices=np.array([[1.0]]),
                    transition_covariance=np.array([[transition_cov]]),
                    observation_covariance=np.array([[observation_cov]]),
                    initial_state_covariance=np.array([[1.0]])
                )
                
                # Apply Kalman Filter
                smoothed_state_means, _ = kf.smooth(data_with_nan.reshape(-1, 1))
                
                # Update imputed data
                if np.any(np.isnan(smoothed_state_means)):
                    random.seed(42)
                    np.random.seed(42)
                    smoothed_state_means = np.clip(np.random.normal(loc=0, scale=10, size=smoothed_state_means.shape), a_min=0, a_max=np.inf)
                imputed[missing_mask] = smoothed_state_means[missing_mask]
            
            # Store the result
            dp_imputed = deepcopy(dp)
            dp_imputed.imputed_history = imputed
            imputed_test_data.append(dp_imputed)
        
        # Save Kalman Filter parameters for this station
        if kalman_params:
            save_path = save_ml_model(kalman_params, "kalman", station_id, output_dir)
            print(f"Saved Kalman Filter parameters for station {station_id} to {save_path}")
    
    return imputed_test_data



def missforest_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint], 
                         logger: logging.Logger, output_dir: str, n_estimators: int=100, max_iter: int=10) -> List[EVChargingDataPoint]:
    """
    MissForest imputation - train on training data and impute test data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points for imputation
        logger: Logger for output
        output_dir: Directory to save trained models
        n_estimators: Number of trees in random forest
        max_iter: Maximum number of iterations
        
    Returns:
        List of imputed test data points
    """
    imputed_test_data = []
    
    try:
        # Try to import MissForest from missingpy
        from missingpy import MissForest
        missforest_available = True
    except ImportError:
        # If not available, use IterativeImputer with RandomForestRegressor
        from sklearn.ensemble import RandomForestRegressor
        missforest_available = False
        logger.warning("missingpy package not found. Using sklearn's IterativeImputer with RandomForestRegressor instead.")
        
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="MissForest imputation"):
        # Get training data for this station (if available)
        train_dps = train_stations_data.get(station_id, [])
        
        # Extract data from test data
        X_test = np.array([dp.history.flatten() for dp in test_dps])
        test_missing_masks = np.array([dp.missing_mask.flatten() for dp in test_dps])
        
        # Replace masked values with NaN in test data
        for i in range(len(X_test)):
            X_test[i][test_missing_masks[i]] = np.nan 
            
        # Check if we have enough training data for proper model fitting
        if len(train_dps) <= 5:  # Not enough training data
            # Fall back to fitting on test data itself
            if len(X_test) <= 5:  # Also not enough test data
                # Fall back to mean imputation for very small datasets
                mean_val = np.nanmean(X_test)
                if np.isnan(mean_val):
                    mean_val = 0.0
                
                for i, dp in enumerate(test_dps):
                    dp_imputed = deepcopy(dp)
                    imputed = dp.history.copy()
                    imputed[dp.missing_mask] = mean_val
                    dp_imputed.imputed_history = imputed
                    imputed_test_data.append(dp_imputed)
                continue
            
            # Fit and transform using just test data
            if missforest_available:
                imputer = MissForest(n_estimators=n_estimators, max_iter=max_iter, random_state=42)
                X_imputed = imputer.fit_transform(X_test)
            else:
                estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                imputer = IterativeImputer(
                    estimator=estimator,
                    max_iter=max_iter,
                    random_state=42,
                    verbose=0
                )
                X_imputed = imputer.fit_transform(X_test)
        else:
            # Extract data from training data for fitting
            X_train = np.array([dp.history.flatten() for dp in train_dps])
            
            # Apply MissForest imputation - fit on train, transform test
            if missforest_available:
                imputer = MissForest(n_estimators=n_estimators, max_iter=max_iter, random_state=42)
                imputer.fit(X_train)
                X_imputed = imputer.transform(X_test)
            else:
                estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                imputer = IterativeImputer(
                    estimator=estimator,
                    max_iter=max_iter,
                    random_state=42,
                    verbose=0
                )
                imputer.fit(X_train)
                X_imputed = imputer.transform(X_test)
            
            # Save the trained imputer
            save_path = save_ml_model(imputer, "missforest", station_id, output_dir)
            logger.info(f"Saved MissForest model for station {station_id} to {save_path}")
        
        # Put imputed values back into test data points
        for i, dp in enumerate(test_dps):
            dp_imputed = deepcopy(dp)
            imputed = X_imputed[i].reshape(dp.history.shape)
            dp_imputed.imputed_history = imputed
            imputed_test_data.append(dp_imputed)
    
    return imputed_test_data




def save_model(model: nn.Module, station_id: int, output_dir: str):
    """
    Save a trained model to disk.
    
    Args:
        model: The PyTorch model to save
        station_id: The station ID for this model
        output_dir: Directory to save model
    """

    
    # Create filename with station_id
    filename = f"station_{station_id}.pt"
    save_path = os.path.join(output_dir, filename)
    
    # Save model
    torch.save(model.state_dict(), save_path)
    
    return save_path


def lstm_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint], 
                   logger: logging.Logger, output_dir: str, hidden_dim: int=64, num_layers: int=2,
                   epochs: int=100, batch_size: int=32, 
                   device: str="cuda") -> List[EVChargingDataPoint]:
    """
    LSTM-based imputation for time series data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points with missing values to impute
        logger: Logger for output messages
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to use for training (cuda/cpu)
        
    Returns:
        List of imputed test data points
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch is required for LSTM imputation. Falling back to mean imputation.")
        return mean_imputation(test_data)
    
    imputed_test_data = []
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Using device: {device} for LSTM imputation")
    
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Define LSTM model
    class LSTMImputer(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super(LSTMImputer, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers, 
                batch_first=True, bidirectional=True
            )
            
            # Output layer
            self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
            
        def forward(self, x):
            # x shape: [batch, seq_len, features]
            batch_size, seq_len = x.size(0), x.size(1)
            
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)  # *2 for bidirectional
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            
            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: [batch, seq_len, hidden_dim * 2]
            
            # Decode hidden states
            out = self.fc(out)
            return out
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="LSTM imputation"):
        # Get training data for this station (if available)
        train_dps = train_stations_data.get(station_id, [])
        
        # Skip if not enough training data
        if len(train_dps) < 5:  # Need reasonable amount for sequence learning
            # Fall back to mean imputation for small datasets
            for dp in test_dps:
                dp_imputed = deepcopy(dp)
                imputed = dp.history.copy()
                # Use mean of observed values
                observed_values = dp.history[~dp.missing_mask]
                mean_val = np.mean(observed_values) if len(observed_values) > 0 else 0.0
                imputed[dp.missing_mask] = mean_val
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
            continue
        
        # Get data dimensions
        seq_length = test_dps[0].history.shape[0]
        output_dim = 1
        if len(test_dps[0].history.shape) > 1:
            output_dim = test_dps[0].history.shape[1]
        
        # Prepare training data for LSTM
        train_full_data = np.array([dp.history for dp in train_dps])
        
        # Prepare test data for LSTM
        test_masked_data = np.array([dp.history for dp in test_dps])
        test_masks = np.array([dp.missing_mask for dp in test_dps])
        
        # Normalize data using training statistics
        data_mean = np.mean(train_full_data.reshape(-1, output_dim), axis=0)
        data_std = np.std(train_full_data.reshape(-1, output_dim), axis=0)
        data_std[data_std == 0] = 1.0  # Avoid division by zero
        
        # Scale data
        train_data_norm = (train_full_data - data_mean) / data_std
        test_masked_norm = (test_masked_data - data_mean) / data_std
        
        # Replace missing values with 0 in test data
        for i in range(len(test_masked_norm)):
            test_masked_norm[i][test_masks[i]] = 0.0
        
        # Convert to torch tensors
        train_data_tensor = torch.tensor(train_data_norm, dtype=torch.float32).to(device)
        test_masked_tensor = torch.tensor(test_masked_norm, dtype=torch.float32).to(device)
        test_masks_tensor = torch.tensor(test_masks, dtype=torch.float32).to(device)
        
        # Create dataset and dataloader for training
        train_dataset = TensorDataset(train_data_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        
        # Initialize model
        model = LSTMImputer(
            input_dim=output_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim,
            num_layers=num_layers
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for (train_seq,) in train_dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(train_seq)
                
                # Compute loss on full training sequence
                loss = criterion(outputs, train_seq)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"[Station {station_id}] [Epoch {epoch+1}/{epochs}] [Loss: {total_loss/len(train_dataloader):.6f}]")
        
        save_path = save_model(model=model, station_id=station_id, output_dir=output_dir)
        logger.info(f"Saved model for station {station_id} to {save_path}")
        
        # Use trained model for imputation
        model.eval()
        with torch.no_grad():
            for i, dp in enumerate(test_dps):
                # Normalize input
                masked_norm = (dp.history - data_mean) / data_std
                masked_norm[dp.missing_mask] = 0.0  # Replace missing with 0
                
                # Convert to tensor
                masked_tensor = torch.tensor(masked_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get model output
                output = model(masked_tensor).squeeze(0).cpu().numpy()
                
                # Denormalize
                output = output * data_std + data_mean
                
                # Combine original and imputed data
                imputed = dp.history.copy()
                imputed[dp.missing_mask] = output[dp.missing_mask]
                
                dp_imputed = deepcopy(dp)
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
    
    return imputed_test_data




def transformer_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint],
                          logger: logging.Logger, output_dir: str, d_model: int=64, nhead: int=4, 
                          num_layers: int=2, epochs: int=100, 
                          batch_size: int=32, device: str="cuda") -> List[EVChargingDataPoint]:
    """
    Transformer-based imputation for time series data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points with missing values to impute
        logger: Logger for output messages
        d_model: Feature dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to use for training (cuda/cpu)
        
    Returns:
        List of imputed test data points
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import math
    except ImportError:
        logger.error("PyTorch is required for Transformer imputation. Falling back to mean imputation.")
        return mean_imputation(test_data)
    
    imputed_test_data = []
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Using device: {device} for Transformer imputation")
    
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Define Transformer model
    class TransformerImputer(nn.Module):
        def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
            super(TransformerImputer, self).__init__()
            
            # Feature embedding
            self.embedding = nn.Linear(input_dim, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model)
            
            # Transformer encoder
            encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
            
            # Output layer
            self.decoder = nn.Linear(d_model, output_dim)
            
        def forward(self, x, src_mask=None):
            # x shape: [batch, seq_len, features]
            x = self.embedding(x)  # [batch, seq_len, d_model]
            x = self.pos_encoder(x)
            
            # Create padding mask (not needed if all sequences are same length)
            if src_mask is None:
                src_mask = torch.ones((x.size(0), x.size(1)), device=x.device).bool()
                
            x = self.transformer_encoder(x)
            return self.decoder(x)
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(PositionalEncoding, self).__init__()
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="Transformer imputation"):
        # Get training data for this station (if available)
        train_dps = train_stations_data.get(station_id, [])
        
        # Skip if not enough training data
        if len(train_dps) < 5:  # Need reasonable amount for attention mechanism
            # Fall back to mean imputation
            for dp in test_dps:
                dp_imputed = deepcopy(dp)
                imputed = dp.history.copy()
                observed_values = dp.history[~dp.missing_mask]
                mean_val = np.mean(observed_values) if len(observed_values) > 0 else 0.0
                imputed[dp.missing_mask] = mean_val
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
            continue
        
        # Get data dimensions
        seq_length = test_dps[0].history.shape[0]
        output_dim = 1
        if len(test_dps[0].history.shape) > 1:
            output_dim = test_dps[0].history.shape[1]
        
        # Prepare training data for Transformer
        train_full_data = np.array([dp.history for dp in train_dps])
        
        # Prepare test data for Transformer
        test_masked_data = np.array([dp.history for dp in test_dps])
        test_masks = np.array([dp.missing_mask for dp in test_dps])
        
        # Normalize data using training statistics
        data_mean = np.mean(train_full_data.reshape(-1, output_dim), axis=0)
        data_std = np.std(train_full_data.reshape(-1, output_dim), axis=0)
        data_std[data_std == 0] = 1.0  # Avoid division by zero
        
        # Scale data
        train_data_norm = (train_full_data - data_mean) / data_std
        test_masked_norm = (test_masked_data - data_mean) / data_std
        
        # Replace missing values with 0 in test data
        for i in range(len(test_masked_norm)):
            test_masked_norm[i][test_masks[i]] = 0.0
        
        # Convert to torch tensors
        train_data_tensor = torch.tensor(train_data_norm, dtype=torch.float32).to(device)
        test_masked_tensor = torch.tensor(test_masked_norm, dtype=torch.float32).to(device)
        test_masks_tensor = torch.tensor(test_masks, dtype=torch.float32).to(device)
        
        # Create dataset and dataloader for training
        train_dataset = TensorDataset(train_data_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        
        # Initialize model
        model = TransformerImputer(
            input_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=output_dim
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for (train_seq,) in train_dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(train_seq)
                
                # Compute loss on full training sequence
                loss = criterion(outputs, train_seq)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"[Station {station_id}] [Epoch {epoch+1}/{epochs}] [Loss: {total_loss/len(train_dataloader):.6f}]")
        
        save_path = save_model(model=model, station_id=station_id, output_dir=output_dir)
        logger.info(f"Saved model for station {station_id} to {save_path}")
        
        # Use trained model for imputation
        model.eval()
        with torch.no_grad():
            for i, dp in enumerate(test_dps):
                # Normalize input
                masked_norm = (dp.history - data_mean) / data_std
                masked_norm[dp.missing_mask] = 0.0  # Replace missing with 0
                
                # Convert to tensor
                masked_tensor = torch.tensor(masked_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get model output
                output = model(masked_tensor).squeeze(0).cpu().numpy()
                
                # Denormalize
                output = output * data_std + data_mean
                
                # Combine original and imputed data
                imputed = dp.history.copy()
                imputed[dp.missing_mask] = output[dp.missing_mask]
                
                dp_imputed = deepcopy(dp)
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
    
    return imputed_test_data




def tcn_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint], 
                  logger: logging.Logger, output_dir: str, num_channels: List[int]=[32, 32, 32],
                  kernel_size: int=3, dropout: float=0.2, epochs: int=100,
                  batch_size: int=32, device: str="cuda") -> List[EVChargingDataPoint]:
    """
    TCN (Temporal Convolutional Network) based imputation for time series data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points with missing values to impute
        logger: Logger for output messages
        num_channels: Number of channels in each layer
        kernel_size: Kernel size for convolutions
        dropout: Dropout rate
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to use for training (cuda/cpu)
        
    Returns:
        List of imputed test data points
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch is required for TCN imputation. Falling back to mean imputation.")
        return mean_imputation(test_data)
    
    imputed_test_data = []
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Using device: {device} for TCN imputation")
    
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Define Causal Conv1d layer
    class CausalConv1d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
            super(CausalConv1d, self).__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, 
                stride=stride, padding=self.padding, dilation=dilation
            )
            
        def forward(self, x):
            x = self.conv(x)
            return x[:, :, :-self.padding]  # Remove padding at the end
    
    # Define TCN Residual Block
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
            super(ResidualBlock, self).__init__()
            
            self.conv1 = CausalConv1d(
                in_channels, out_channels, kernel_size, 
                stride=stride, dilation=dilation
            )
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            
            self.conv2 = CausalConv1d(
                out_channels, out_channels, kernel_size, 
                stride=stride, dilation=dilation
            )
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
            self.relu = nn.ReLU()
            
        def forward(self, x):
            residual = x
            
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.dropout1(out)
            
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.dropout2(out)
            
            if self.downsample is not None:
                residual = self.downsample(residual)
            
            return self.relu(out + residual)
    
    # Define TCN Model
    class TCN(nn.Module):
        def __init__(self, input_dim, output_dim, num_channels, kernel_size, dropout):
            super(TCN, self).__init__()
            
            self.input_projection = nn.Conv1d(input_dim, num_channels[0], 1)
            
            layers = []
            for i in range(len(num_channels) - 1):
                dilation = 2 ** i  # Exponentially increasing dilation
                layers.append(ResidualBlock(
                    num_channels[i], num_channels[i+1], kernel_size, stride=1,
                    dilation=dilation, dropout=dropout
                ))
            
            self.network = nn.Sequential(*layers)
            self.output_projection = nn.Conv1d(num_channels[-1], output_dim, 1)
            
        def forward(self, x):
            # x shape: [batch, seq_len, features]
            x = x.transpose(1, 2)  # [batch, features, seq_len]
            
            x = self.input_projection(x)
            x = self.network(x)
            x = self.output_projection(x)
            
            return x.transpose(1, 2)  # [batch, seq_len, features]
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="TCN imputation"):
        # Get training data for this station (if available)
        train_dps = train_stations_data.get(station_id, [])
        
        # Skip if not enough training data
        if len(train_dps) < 5:  # Need reasonable amount for convolutions
            # Fall back to mean imputation
            for dp in test_dps:
                dp_imputed = deepcopy(dp)
                imputed = dp.history.copy()
                observed_values = dp.history[~dp.missing_mask]
                mean_val = np.mean(observed_values) if len(observed_values) > 0 else 0.0
                imputed[dp.missing_mask] = mean_val
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
            continue
        
        # Get data dimensions
        seq_length = test_dps[0].history.shape[0]
        output_dim = 1
        if len(test_dps[0].history.shape) > 1:
            output_dim = test_dps[0].history.shape[1]
        
        # Prepare training data for TCN
        train_full_data = np.array([dp.history for dp in train_dps])
        
        # Prepare test data for TCN
        test_masked_data = np.array([dp.history for dp in test_dps])
        test_masks = np.array([dp.missing_mask for dp in test_dps])
        
        # Normalize data using training statistics
        data_mean = np.mean(train_full_data.reshape(-1, output_dim), axis=0)
        data_std = np.std(train_full_data.reshape(-1, output_dim), axis=0)
        data_std[data_std == 0] = 1.0  # Avoid division by zero
        
        # Scale data
        train_data_norm = (train_full_data - data_mean) / data_std
        test_masked_norm = (test_masked_data - data_mean) / data_std
        
        # Replace missing values with 0 in test data
        for i in range(len(test_masked_norm)):
            test_masked_norm[i][test_masks[i]] = 0.0
        
        # Convert to torch tensors
        train_data_tensor = torch.tensor(train_data_norm, dtype=torch.float32).to(device)
        test_masked_tensor = torch.tensor(test_masked_norm, dtype=torch.float32).to(device)
        test_masks_tensor = torch.tensor(test_masks, dtype=torch.float32).to(device)
        
        # Create dataset and dataloader for training
        train_dataset = TensorDataset(train_data_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        
        # Initialize model
        model = TCN(
            input_dim=output_dim,
            output_dim=output_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for (train_seq,) in train_dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(train_seq)
                
                # Compute loss on full training sequence
                loss = criterion(outputs, train_seq)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"[Station {station_id}] [Epoch {epoch+1}/{epochs}] [Loss: {total_loss/len(train_dataloader):.6f}]")
        
        
        save_path = save_model(model=model, station_id=station_id, output_dir=output_dir)
        logger.info(f"Saved model for station {station_id} to {save_path}")
        
        
        # Use trained model for imputation
        model.eval()
        with torch.no_grad():
            for i, dp in enumerate(test_dps):
                # Normalize input
                masked_norm = (dp.history - data_mean) / data_std
                masked_norm[dp.missing_mask] = 0.0  # Replace missing with 0
                
                # Convert to tensor
                masked_tensor = torch.tensor(masked_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get model output
                output = model(masked_tensor).squeeze(0).cpu().numpy()
                
                # Denormalize
                output = output * data_std + data_mean
                
                # Combine original and imputed data
                imputed = dp.history.copy()
                imputed[dp.missing_mask] = output[dp.missing_mask]
                
                dp_imputed = deepcopy(dp)
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
    
    return imputed_test_data






def gan_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint],
                  logger: logging.Logger, output_dir: str, latent_dim: int=64, epochs: int=100, 
                  batch_size: int=32, device: str="cuda") -> List[EVChargingDataPoint]:
    """
    GAN imputation - train on training data and impute test data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points with missing values to impute
        logger: Logger for output messages
        latent_dim: Dimension of latent space
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to use for training (cuda/cpu)
        
    Returns:
        List of imputed test data points
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch is required for GAN imputation. Falling back to mean imputation.")
        return mean_imputation(test_data)
    
    imputed_test_data = []
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Using device: {device} for GAN imputation")
    
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Define GAN models
    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, output_dim),
            )
        
        def forward(self, x):
            return self.model(x)
    
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.model(x)
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="GAN imputation"):
        # Get training data for this station
        train_dps = train_stations_data.get(station_id, [])
        
        # Skip if not enough training data
        if len(train_dps) < batch_size:
            # Fall back to mean imputation for small datasets
            for dp in test_dps:
                dp_imputed = deepcopy(dp)
                imputed = dp.history.copy()
                # Use mean of observed values
                observed_values = dp.history[~dp.missing_mask]
                mean_val = np.mean(observed_values) if len(observed_values) > 0 else 0.0
                imputed[dp.missing_mask] = mean_val
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
            continue
            
        # Extract data dimensions
        seq_length = train_dps[0].history.shape[0]
        flat_length = np.prod(train_dps[0].history.shape)
        
        # Prepare training data for GAN
        train_full_data = np.array([dp.history.flatten() for dp in train_dps])
        
        # Prepare test data for GAN
        test_masked_data = np.array([dp.history.flatten() for dp in test_dps])
        test_masks = np.array([dp.missing_mask.flatten() for dp in test_dps])
        
        # Normalize data using training statistics
        data_mean = np.mean(train_full_data, axis=0)
        data_std = np.std(train_full_data, axis=0)
        data_std[data_std == 0] = 1.0  # Avoid division by zero
        
        # Scale data
        train_data_norm = (train_full_data - data_mean) / data_std
        test_masked_norm = (test_masked_data - data_mean) / data_std
        
        # Replace missing values with 0 in test data
        for i in range(len(test_masked_norm)):
            test_masked_norm[i][test_masks[i]] = 0.0
        
        # Convert training data to torch tensors
        train_data_tensor = torch.tensor(train_data_norm, dtype=torch.float32).to(device)
        
        # Create dataset and dataloader for training
        train_dataset = TensorDataset(train_data_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        
        # Initialize models
        generator = Generator(latent_dim + flat_length, flat_length).to(device)
        discriminator = Discriminator(flat_length).to(device)
        
        # Loss and optimizers
        adversarial_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()
        
        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Training loop
        for epoch in range(epochs):
            total_g_loss = 0
            total_d_loss = 0
            
            for (real_batch,) in train_dataloader:
                batch_size = real_batch.size(0)
                
                # Create corrupted version of real data for training (random masking)
                mask = torch.bernoulli(torch.ones(batch_size, flat_length) * 0.2).to(device)  # 20% random masking
                masked = real_batch.clone()
                masked = masked * (1 - mask)  # Apply mask
                
                # Labels for real and fake data
                valid = torch.ones(batch_size, 1).to(device)
                fake = torch.zeros(batch_size, 1).to(device)
                
                # Train Generator
                optimizer_G.zero_grad()
                
                # Generate random noise
                z = torch.randn(batch_size, latent_dim).to(device)
                gen_input = torch.cat([z, masked], dim=1)
                
                # Generate imputed data
                gen_output = generator(gen_input)
                
                # Combine real and generated data based on mask
                combined = masked.clone()
                combined[mask == 1] = gen_output[mask == 1]
                
                # Compute loss
                validity = discriminator(combined)
                rec_loss = l1_loss(combined * mask, real_batch * mask)
                g_loss = 0.1 * adversarial_loss(validity, valid) + 0.9 * rec_loss
                
                g_loss.backward()
                optimizer_G.step()
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                # Compute loss for real data
                real_validity = discriminator(real_batch)
                real_loss = adversarial_loss(real_validity, valid)
                
                # Compute loss for fake data
                fake_validity = discriminator(combined.detach())
                fake_loss = adversarial_loss(fake_validity, fake)
                
                d_loss = 0.5 * (real_loss + fake_loss)
                
                d_loss.backward()
                optimizer_D.step()
                
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"[Station {station_id}] [Epoch {epoch+1}/{epochs}] [G loss: {total_g_loss/len(train_dataloader):.4f}] [D loss: {total_d_loss/len(train_dataloader):.4f}]")
        
        
        # Save Generator model
        gen_save_dir = os.path.join(output_dir, "gan_generator")
        os.makedirs(gen_save_dir, exist_ok=True)
        gen_filename = f"generator_station_{station_id}.pt"
        gen_save_path = os.path.join(gen_save_dir, gen_filename)
        torch.save(generator.state_dict(), gen_save_path)
        logger.info(f"Saved GAN Generator model for station {station_id} to {gen_save_path}")

        # Save Discriminator model (optional but recommended)
        disc_save_dir = os.path.join(output_dir, "gan_discriminator")
        os.makedirs(disc_save_dir, exist_ok=True)
        disc_filename = f"discriminator_station_{station_id}.pt"
        disc_save_path = os.path.join(disc_save_dir, disc_filename)
        torch.save(discriminator.state_dict(), disc_save_path)
        logger.info(f"Saved GAN Discriminator model for station {station_id} to {disc_save_path}")
        
        
        # Use trained GAN for imputation on test data
        generator.eval()
        with torch.no_grad():
            for i, dp in enumerate(test_dps):
                # Normalize input
                masked_norm = (dp.history.flatten() - data_mean) / data_std
                masked_norm[dp.missing_mask.flatten()] = 0.0  # Replace missing with 0
                
                # Convert to tensor
                masked_tensor = torch.tensor(masked_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Generate random noise for imputation
                z = torch.randn(1, latent_dim).to(device)
                gen_input = torch.cat([z, masked_tensor], dim=1)
                
                # Generate imputed data
                gen_output = generator(gen_input).squeeze(0).cpu().numpy()
                
                # Denormalize
                gen_output = gen_output * data_std + data_mean
                
                # Combine original and imputed data
                imputed = dp.history.copy()
                imputed[dp.missing_mask] = gen_output[dp.missing_mask.flatten()].reshape(dp.missing_mask.sum())
                
                dp_imputed = deepcopy(dp)
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
    
    return imputed_test_data



def autoencoder_imputation(train_data: List[EVChargingDataPoint], test_data: List[EVChargingDataPoint],
                          logger: logging.Logger, output_dir: str, hidden_dim: int=64, epochs: int=100, 
                          batch_size: int=32, device: str="cuda") -> List[EVChargingDataPoint]:
    """
    Denoising Autoencoder imputation - train on training data and impute test data.
    
    Args:
        train_data: List of data points for training
        test_data: List of data points with missing values to impute
        logger: Logger for output messages
        hidden_dim: Hidden dimension size
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to use for training (cuda/cpu)
        
    Returns:
        List of imputed test data points
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch is required for Autoencoder imputation. Falling back to mean imputation.")
        return mean_imputation(test_data)
    
    imputed_test_data = []
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Using device: {device} for Autoencoder imputation")
    
    # Group data points by station_id
    train_stations_data = {}
    test_stations_data = {}
    
    # Organize training data by station
    for dp in train_data:
        if dp.station_id not in train_stations_data:
            train_stations_data[dp.station_id] = []
        train_stations_data[dp.station_id].append(dp)
        
    # Organize test data by station
    for dp in test_data:
        if dp.station_id not in test_stations_data:
            test_stations_data[dp.station_id] = []
        test_stations_data[dp.station_id].append(dp)
    
    # Define Autoencoder model
    class DenoisingAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(DenoisingAutoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, input_dim)
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    # Process each station's data
    for station_id, test_dps in tqdm(test_stations_data.items(), desc="Autoencoder imputation"):
        # Get training data for this station
        train_dps = train_stations_data.get(station_id, [])
        
        # Skip if not enough training data
        if len(train_dps) < batch_size:
            # Fall back to mean imputation for small datasets
            for dp in test_dps:
                dp_imputed = deepcopy(dp)
                imputed = dp.history.copy()
                # Use mean of observed values
                observed_values = dp.history[~dp.missing_mask]
                mean_val = np.mean(observed_values) if len(observed_values) > 0 else 0.0
                imputed[dp.missing_mask] = mean_val
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
            continue
            
        # Extract data dimensions
        flat_length = np.prod(train_dps[0].history.shape)
        
        # Prepare training data for autoencoder
        train_full_data = np.array([dp.history.flatten() for dp in train_dps])
        
        # Prepare test data for autoencoder
        test_masked_data = np.array([dp.history.flatten() for dp in test_dps])
        test_masks = np.array([dp.missing_mask.flatten() for dp in test_dps])
        
        # Normalize data using training statistics
        data_mean = np.mean(train_full_data, axis=0)
        data_std = np.std(train_full_data, axis=0)
        data_std[data_std == 0] = 1.0  # Avoid division by zero
        
        # Scale data
        train_data_norm = (train_full_data - data_mean) / data_std
        test_masked_norm = (test_masked_data - data_mean) / data_std
        
        # Replace missing values with 0 in test data
        for i in range(len(test_masked_norm)):
            test_masked_norm[i][test_masks[i]] = 0.0
        
        # Convert to torch tensors
        train_data_tensor = torch.tensor(train_data_norm, dtype=torch.float32).to(device)
        test_masked_tensor = torch.tensor(test_masked_norm, dtype=torch.float32).to(device)
        test_masks_tensor = torch.tensor(test_masks, dtype=torch.float32).to(device)
        
        # Create dataset and dataloader for training
        train_dataset = TensorDataset(train_data_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        
        # Initialize model
        model = DenoisingAutoencoder(input_dim=flat_length, hidden_dim=hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Generate corrupted versions of training data for denoising training
        def add_noise(data_batch, noise_factor=0.2):
            # Create random mask with 20% of values to corrupt
            mask = torch.bernoulli(torch.ones(data_batch.size()) * noise_factor).to(device)
            # Add random noise to masked values
            noisy_data = data_batch.clone()
            noise = torch.randn_like(data_batch)
            noisy_data[mask == 1] = noise[mask == 1]
            return noisy_data, mask
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for (clean_batch,) in train_dataloader:
                # Create noisy version for denoising training
                noisy_batch, noise_mask = add_noise(clean_batch)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(noisy_batch)
                
                # Compute reconstruction loss
                # Focus more on the noisy parts that need to be reconstructed
                loss = criterion(outputs, clean_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"[Station {station_id}] [Epoch {epoch+1}/{epochs}] [Loss: {total_loss/len(train_dataloader):.6f}]")
        
        save_path = save_model(model=model, station_id=station_id, output_dir=output_dir)
        logger.info(f"Saved model for station {station_id} to {save_path}")
        
        # Use trained autoencoder for imputation on test data
        model.eval()
        with torch.no_grad():
            for i, dp in enumerate(test_dps):
                # Normalize input
                masked_norm = (dp.history.flatten() - data_mean) / data_std
                masked_norm[dp.missing_mask.flatten()] = 0.0  # Replace missing with 0
                
                # Convert to tensor
                masked_tensor = torch.tensor(masked_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get model output
                output = model(masked_tensor).squeeze(0).cpu().numpy()
                
                # Denormalize
                output = output * data_std + data_mean
                
                # Combine original and imputed data
                imputed = dp.history.copy()
                imputed[dp.missing_mask] = output[dp.missing_mask.flatten()].reshape(dp.missing_mask.sum())
                
                dp_imputed = deepcopy(dp)
                dp_imputed.imputed_history = imputed
                imputed_test_data.append(dp_imputed)
    
    return imputed_test_data