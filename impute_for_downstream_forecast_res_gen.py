import os
import argparse
import pandas as pd
import numpy as np
import json
import logging
import torch 
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import NUM_STATIONS

# NeuralForecast imports
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, TiDE, DLinear, PatchTST, TSMixer, iTransformer, TimesNet

torch.set_default_device('cpu')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("downstream_forecast.log"), logging.StreamHandler()]
)
logger = logging.getLogger("downstream_forecast")


class DownstreamForecastEvaluator:
    """Evaluate imputation methods on downstream forecasting performance."""
    
    def __init__(self, city: str, horizon: int = 7, freq: str = 'D'):
        """
        Initialize the evaluator.
        
        Args:
            city: City name
            horizon: Forecast horizon (days)
            freq: Frequency of the data ('D' for daily)
        """
        self.city = city
        self.horizon = horizon
        self.freq = freq
        self.models = self._init_models()
        
        
    def _init_models(self) -> List:
        """Initialize forecasting models with recent state-of-the-art models."""
        models = [
            # Original models
            NBEATS(
                input_size=self.horizon * 2,
                h=self.horizon,
                max_steps=50,
                scaler_type='robust',
                enable_progress_bar=True,
                accelerator='cpu'
            ),
            NHITS(
                input_size=self.horizon * 2,
                h=self.horizon,
                max_steps=50,
                scaler_type='robust',
                enable_progress_bar=True,
                accelerator='cpu'
            ),
            TiDE(
                input_size=self.horizon * 2,
                h=self.horizon,
                max_steps=50,
                scaler_type='robust',
                enable_progress_bar=True,
                accelerator='cpu'
            ),
            DLinear(
                input_size=self.horizon * 2,
                h=self.horizon,
                max_steps=50,
                scaler_type='robust',
                enable_progress_bar=True,
                accelerator='cpu'
            ),
            # # New state-of-the-art models
            # PatchTST(
            #     input_size=self.horizon * 2,  # Increased input size
            #     h=self.horizon,
            #     patch_len=1,      # Smaller patch length
            #     stride=1,         # Stride for patching
            #     hidden_size=128,  # Hidden layer size
            #     n_heads=4,        # Number of attention heads
            #     max_steps=50,
            #     scaler_type='robust',
            #     enable_progress_bar=True,
            #     accelerator='cpu'
            # ),
            TSMixer(
                input_size=self.horizon * 2,
                n_series=NUM_STATIONS[self.city],
                h=self.horizon,
                n_block=2,    # Number of mixing blocks
                max_steps=50,
                scaler_type='robust',
                enable_progress_bar=True,
                accelerator='cpu'
            ),
            iTransformer(
                input_size=self.horizon * 2,
                n_series=NUM_STATIONS[self.city],
                h=self.horizon,
                n_heads=4,     # Number of attention heads,
                max_steps=50,
                scaler_type='robust',
                enable_progress_bar=True,
                accelerator='cpu'
            ),
            TimesNet(
                input_size=self.horizon * 2,
                h=self.horizon,
                top_k=3,      # Top-k frequencies for TimesBlock
                max_steps=50,
                scaler_type='robust',
                enable_progress_bar=True,
                accelerator='cpu'
            )
        ]
        return models
    
    
    def load_original_complete_data(self, data_dir: str) -> pd.DataFrame:
        """
        Load original complete data without any artificial missing values.
        
        Args:
            data_dir: Directory containing the original data
            
        Returns:
            DataFrame with complete charging data
        """
        logger.info(f"Loading original complete data for {self.city}")
        
        # Load the original daily data without NaNs
        daily_data_path = os.path.join(data_dir, "daily_data_withNaNs.csv")
        if not os.path.exists(daily_data_path):
            raise FileNotFoundError(f"Original data not found at {daily_data_path}")
        
        # Load data
        if self.city == 'PaloAlto':
            df = pd.read_csv(daily_data_path, parse_dates=["start_date"])
        else:
            df = pd.read_csv(daily_data_path, parse_dates=["start_date"], index_col=[0])
        if self.city != 'PaloAlto':
            df.set_index('start_date', inplace=True)
            df = df.reset_index()
        
        # Reshape data to long format for NeuralForecast
        df_long = df.melt(
            id_vars=['start_date'], 
            var_name='station_id', 
            value_name='y'
        )
        df_long.rename(columns={'start_date': 'ds'}, inplace=True)
        
        # Remove rows with NaN values (natural missing data)
        df_long = df_long.dropna().reset_index(drop=True)
        
        # Ensure proper data types
        df_long['ds'] = pd.to_datetime(df_long['ds'])
        df_long['y'] = df_long['y'].astype(float)
        
        # Create unique_id column (required by NeuralForecast)
        df_long['unique_id'] = df_long['station_id'].astype(str)
        
        # Sort by station and date
        df_long = df_long.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df_long)} data points from {len(df_long['unique_id'].unique())} stations")
        
        return df_long[['unique_id', 'ds', 'y']]
    
    
    def load_imputed_data_with_masks(self, imputation_results_dir: str, method: str) -> Tuple[pd.DataFrame, Set[Tuple[str, str]]]:
        """
        Load imputed data from imputation results and track which points are observed vs imputed.
        
        Args:
            imputation_results_dir: Directory containing imputation results
            method: Imputation method name
            
        Returns:
            Tuple of (DataFrame with imputed charging data, Set of (station_id, date) tuples for observed data)
        """
        logger.info(f"Loading imputed data for method: {method}")
        
        # Load detailed records from imputation results
        records_file = os.path.join(imputation_results_dir, f"{method}_{self.city}_detailed_records.json")
        if not os.path.exists(records_file):
            raise FileNotFoundError(f"Imputation records not found at {records_file}")
        
        with open(records_file, 'r') as f:
            records = json.load(f)
        
        # Convert to DataFrame
        df_records = pd.DataFrame(records)
        
        # Create a complete dataset by combining observed and imputed values
        # Also track which points are observed (ground truth)
        df_long = []
        observed_points = set()
        
        for _, record in df_records.iterrows():
            station_id = str(record['station_id'])
            date_str = str(pd.to_datetime(record['date']).date())
            
            if record['value_type'] == 'observed':
                value = record['observed_value']
                # Mark this point as observed (ground truth)
                observed_points.add((station_id, date_str))
            else:  # imputed
                value = record['imputed_value']
                # This point is imputed, not ground truth
            
            df_long.append({
                'unique_id': station_id,
                'ds': pd.to_datetime(record['date']),
                'y': float(value)
            })
        
        df_long = pd.DataFrame(df_long)
        
        # Sort by station and date
        df_long = df_long.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df_long)} imputed data points from {len(df_long['unique_id'].unique())} stations")
        logger.info(f"Ground truth points: {len(observed_points)}, Imputed points: {len(df_long) - len(observed_points)}")
        
        return df_long, observed_points
    
    
    def prepare_forecast_data(self, df: pd.DataFrame, min_length: int = None) -> pd.DataFrame:
        """
        Prepare data for forecasting by ensuring sufficient history length.
        
        Args:
            df: Input DataFrame
            min_length: Minimum required length per station
            
        Returns:
            Filtered DataFrame ready for forecasting
        """
        if min_length is None:
            min_length = self.horizon * 3  # Need at least 3x horizon for train/val/test
        
        # Filter stations with sufficient data
        station_counts = df.groupby('unique_id').size()
        valid_stations = station_counts[station_counts >= min_length].index
        
        df_filtered = df[df['unique_id'].isin(valid_stations)].copy()
        
        logger.info(f"Filtered to {len(valid_stations)} stations with at least {min_length} data points")
        
        return df_filtered
    
    
    def create_train_test_split(self, df: pd.DataFrame, train_size: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split for forecasting evaluation.
        
        Args:
            df: Input DataFrame
            test_size: Size of test set (default: horizon)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        
        train_dfs = []
        test_dfs = []
        
        for station_id in df['unique_id'].unique():
            station_data = df[df['unique_id'] == station_id].sort_values('ds').reset_index(drop=True)
            train_size = int(len(station_data) * 0.8)
            
            if len(station_data) < train_size + self.horizon:
                continue  # Skip stations with insufficient data
            
            # Split into train and test
            train_data = station_data.iloc[:train_size].copy()
            test_data = station_data.iloc[train_size:].copy()
            
            train_dfs.append(train_data)
            test_dfs.append(test_data)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        logger.info(f"Created train/test split: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df
    

    def evaluate_forecasting_performance(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                    observed_points: Set[Tuple[str, str]] = None,
                                    model_save_dir: str = None, method_name: str = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate forecasting performance using multiple models.
        Only calculate metrics on ground truth data points (not imputed values).
        
        Args:
            train_df: Training data
            test_df: Test data
            observed_points: Set of (station_id, date) tuples that are ground truth (not imputed)
            model_save_dir: Directory to save trained models
            method_name: Name of the method (for model saving)
            
        Returns:
            Dictionary with performance metrics for each model
        """
        results = {}
        
        # Initialize NeuralForecast
        nf = NeuralForecast(models=self.models, freq=self.freq)
        
        # Fit models
        logger.info("Training forecasting models...")
        nf.fit(train_df)
        
        # Save trained models if directory is provided
        os.makedirs(model_save_dir, exist_ok=True)
        model_file = os.path.join(model_save_dir, f"{self.city}_{method_name}_neuralforecast_models.pkl")
        
        # Save the entire NeuralForecast object
        import pickle
        with open(model_file, 'wb') as f:
            pickle.dump(nf, f)
        logger.info(f"Saved trained models to {model_file}")
        
        # Also save model metadata
        metadata = {
            'city': self.city,
            'method': method_name,
            'horizon': self.horizon,
            'freq': self.freq,
            'models': [model.__class__.__name__ for model in self.models],
            'training_data_shape': train_df.shape,
            'n_stations': len(train_df['unique_id'].unique()),
            'training_timestamp': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(model_save_dir, f"{self.city}_{method_name}_model_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to {metadata_file}")

        
        # Generate forecasts
        logger.info("Generating forecasts...")
        forecasts = nf.predict()
        
        # Merge with test data for evaluation
        evaluation_df = test_df.merge(
            forecasts, 
            on=['unique_id', 'ds'], 
            how='inner'
        )
        
        # If observed_points is provided, filter to only ground truth predictions
        if observed_points is not None:
            # Create a mask for ground truth points
            evaluation_df['date_str'] = evaluation_df['ds'].dt.date.astype(str)
            evaluation_df['is_ground_truth'] = evaluation_df.apply(
                lambda row: (row['unique_id'], row['date_str']) in observed_points, 
                axis=1
            )
            
            # Filter to only ground truth points
            original_len = len(evaluation_df)
            evaluation_df = evaluation_df[evaluation_df['is_ground_truth']].copy()
            ground_truth_len = len(evaluation_df)
            
            logger.info(f"Filtered evaluation data: {original_len} total predictions -> {ground_truth_len} ground truth predictions")
            
            if ground_truth_len == 0:
                logger.warning("No ground truth predictions found in test set")
                return {}
        
        # Calculate metrics for each model
        for model in self.models:
            model_name = model.__class__.__name__
            
            if model_name in evaluation_df.columns:
                y_true = evaluation_df['y'].values
                y_pred = evaluation_df[model_name].values
                
                # Remove any NaN values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                
                if len(y_true) > 0:
                    # Calculate metrics
                    mae = np.mean(np.abs(y_true - y_pred))
                    mse = np.mean((y_true - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    
                    # Avoid division by zero for MAPE
                    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
                    
                    results[model_name] = {
                        'MAE': float(mae),
                        'MSE': float(mse),
                        'RMSE': float(rmse),
                        'MAPE': float(mape),
                        'n_predictions': len(y_true),
                        'evaluation_type': 'ground_truth_only' if observed_points is not None else 'all_predictions'
                    }
                    
                    eval_type = "ground truth only" if observed_points is not None else "all predictions"
                    logger.info(f"{model_name} ({eval_type}) - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, N: {len(y_true)}")
                else:
                    logger.warning(f"No valid predictions for {model_name}")
            else:
                logger.warning(f"Model {model_name} not found in forecast results")
                

        
        return results
    
    


    def run_evaluation(self, original_data_dir: str, imputation_results_dir: str, 
                    methods: List[str], output_dir: str) -> Dict[str, Any]:
        """
        Run complete evaluation comparing original vs imputed data forecasting.
        
        Args:
            original_data_dir: Directory with original complete data
            imputation_results_dir: Directory with imputation results
            methods: List of imputation methods to evaluate
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectory for trained models
        model_save_dir = os.path.join(output_dir, 'trained_models')
        os.makedirs(model_save_dir, exist_ok=True)
        
        results = {
            'city': self.city,
            'horizon': self.horizon,
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_note': 'Metrics calculated only on ground truth data points (not imputed values)',
            'model_save_directory': model_save_dir,
            'methods': {}
        }
        
        # Load original complete data once to use as ground truth reference
        logger.info("Loading original complete data for ground truth reference")
        original_df = self.load_original_complete_data(original_data_dir)
        original_df = self.prepare_forecast_data(original_df)
        
        # Create ground truth set from original data
        original_ground_truth = set()
        for _, row in original_df.iterrows():
            date_str = str(row['ds'].date())
            original_ground_truth.add((row['unique_id'], date_str))
        
        logger.info(f"Ground truth reference contains {len(original_ground_truth)} data points")
        
        # Evaluate original complete data as baseline
        logger.info("Evaluating original complete data (baseline)")
        train_df, test_df = self.create_train_test_split(original_df)
        
        # For original data, all test points are ground truth
        test_ground_truth = set()
        for _, row in test_df.iterrows():
            date_str = str(row['ds'].date())
            test_ground_truth.add((row['unique_id'], date_str))
        
        original_results = self.evaluate_forecasting_performance(
            train_df, test_df, test_ground_truth, 
            model_save_dir=model_save_dir, method_name='original_complete'
        )
        
        results['methods']['original_complete'] = {
            'description': 'Original complete data without any missing values',
            'performance': original_results,
            'data_stats': {
                'n_stations': len(original_df['unique_id'].unique()),
                'n_total_points': len(original_df),
                'n_train_points': len(train_df),
                'n_test_points': len(test_df),
                'n_ground_truth_test_points': len(test_ground_truth)
            }
        }
        
        # Evaluate each imputation method
        for method in methods:
            logger.info(f"Evaluating imputation method: {method}")
            
            # Load imputed data with ground truth masks
            imputed_df, observed_points = self.load_imputed_data_with_masks(imputation_results_dir, method)
            imputed_df = self.prepare_forecast_data(imputed_df)
            
            train_df, test_df = self.create_train_test_split(imputed_df)
            
            # Filter observed_points to only include test set points
            test_observed_points = set()
            for _, row in test_df.iterrows():
                date_str = str(row['ds'].date())
                station_date = (row['unique_id'], date_str)
                if station_date in observed_points:
                    test_observed_points.add(station_date)
            
            # Evaluate performance only on ground truth test points
            imputed_results = self.evaluate_forecasting_performance(
                train_df, test_df, test_observed_points,
                model_save_dir=model_save_dir, method_name=method
            )
            
            results['methods'][method] = {
                'description': f'Data imputed using {method} method',
                'performance': imputed_results,
                'data_stats': {
                    'n_stations': len(imputed_df['unique_id'].unique()),
                    'n_total_points': len(imputed_df),
                    'n_train_points': len(train_df),
                    'n_test_points': len(test_df),
                    'n_ground_truth_test_points': len(test_observed_points),
                    'n_imputed_test_points': len(test_df) - len(test_observed_points)
                }
            }
            
            logger.info(f"Method {method}: {len(test_observed_points)} ground truth test points, "
                        f"{len(test_df) - len(test_observed_points)} imputed test points")
                

            imputed_df.to_csv(os.path.join(output_dir, f'{self.city}_imputed.csv'), index=False)
        
        # Calculate relative improvements
        if 'original_complete' in results['methods']:
            results['relative_improvements'] = self._calculate_relative_improvements(results)
        
        # Save results
        results_file = os.path.join(output_dir, f"{self.city}_downstream_forecast_evaluation.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_file}")
        logger.info(f"Trained models saved to {model_save_dir}")
        
        return results
  
    def _calculate_relative_improvements(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate relative improvements of imputation methods compared to original data.
        
        Args:
            results: Complete evaluation results
            
        Returns:
            Dictionary with relative improvements
        """
        improvements = {}
        original_performance = results['methods']['original_complete']['performance']
        
        for method, method_results in results['methods'].items():
            if method == 'original_complete' or 'error' in method_results:
                continue
            
            method_performance = method_results.get('performance', {})
            improvements[method] = {}
            
            for model_name in original_performance.keys():
                if model_name in method_performance:
                    improvements[method][model_name] = {}
                    
                    for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                        if metric in original_performance[model_name] and metric in method_performance[model_name]:
                            original_value = original_performance[model_name][metric]
                            imputed_value = method_performance[model_name][metric]
                            
                            # Calculate relative improvement (positive means better)
                            if original_value > 0:
                                improvement = (original_value - imputed_value) / original_value * 100
                                improvements[method][model_name][metric] = improvement
        
        return improvements


def main():
    """Main function to run downstream forecasting evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate imputation methods on downstream forecasting performance")
    
    # Basic parameters
    parser.add_argument('--city', type=str, default='Boulder', 
                       choices=['PaloAlto', 'Boulder', 'Dundee', 'Perth'],
                       help='City to evaluate')
    parser.add_argument('--horizon', type=int, default=3, 
                       help='Forecast horizon in days')
    parser.add_argument('--freq', type=str, default='D', 
                       help='Data frequency (D for daily)')
    
    # Data directories
    parser.add_argument('--original_data_dir', type=str, default=None,
                       help='Directory containing original complete data')
    parser.add_argument('--imputation_results_dir', type=str, default='res_PRAIM_impute',
                       help='Directory containing imputation results')
    parser.add_argument('--output_dir', type=str, default='res_downstream_forecast',
                       help='Directory to save evaluation results')
    
    # Methods to evaluate
    parser.add_argument('--methods', type=str, nargs='+', default=['PRAIM'],
                       help='Imputation methods to evaluate')
    
    args = parser.parse_args()
    
    # Set default original data directory if not provided
    if args.original_data_dir is None:
        args.original_data_dir = f'data_{args.city}'
    
    # Initialize evaluator
    evaluator = DownstreamForecastEvaluator(
        city=args.city,
        horizon=args.horizon,
        freq=args.freq
    )
    
    # Run evaluation
    logger.info(f"Starting downstream forecasting evaluation for {args.city}")
    logger.info(f"Methods to evaluate: {args.methods}")
    
    results = evaluator.run_evaluation(
        original_data_dir=args.original_data_dir,
        imputation_results_dir=args.imputation_results_dir,
        methods=args.methods,
        output_dir=args.output_dir
    )
    
    logger.info("Downstream forecasting evaluation completed successfully")


if __name__ == "__main__":
    main()