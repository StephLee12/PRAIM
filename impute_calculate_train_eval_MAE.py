import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()]
)
logger = logging.getLogger("evaluate.py")


# Define the order for plotting
BASELINE_ORDER = [
    'PRAIM',
    # Statistical methods
    'mean', 'zero', 'last_observed', 'interpolation',
    # ML methods
    'knn', 'svd', 'kalman', 'missforest',
    # DL methods
    'lstm', 'transformer', 'tcn', 'gan', 'autoencoder'
]

# List of valid baseline models from run_baseline.py
VALID_BASELINES = [
    "mean", "zero", "last_observed", "interpolation", 
    "knn", "svd", "kalman", "missforest", 
    "lstm", "transformer", "tcn", "gan", "autoencoder"
]

BASELINE_MAPPING = {
    'PRAIM': 'PRAIM',
    'mean': 'Mean', 'zero': 'Zero', 'last_observed': 'LO', 'interpolation': 'IP',
    'knn': 'KNN', 'svd': 'MaF', 'kalman': 'KF', 'missforest': 'MF', 
    'lstm': 'LSTM', 'transformer': 'TF', 'tcn': 'TCN', 'gan': 'GAN', 'autoencoder': 'DAE'
}

# All available metrics (excluding MAPE as requested)
METRICS = ['imputation_mae']

CITIES = ['PaloAlto', 'Boulder', 'Dundee', 'Perth']
MISSING_RATIOS = [0.2, 0.3, 0.5, 0.6, 0.8, 0.9]




def find_PRAIM_results(base_dir: str='outputs', cities=None, window_size: int=7, 
                       poi_radius: int=2000, missing_ratios=None) -> list:
    """
    Find all PRAIM model result files based on filtering criteria.
    
    Args:
        base_dir: Base directory for outputs
        cities: List of cities to include (None means all)
        window_size: Window size to filter for (default: 7)
        poi_radius: POI radius to filter for (default: 2000)
        missing_ratios: List of missing ratios to filter for (None means [0.2-0.9])
        
    Returns:
        List of tuples with (city, window_size, poi_radius, missing_ratio, file_path)
    """
    result_files = []
    
    # Default values if none provided
    if cities is None:
        cities = CITIES
        
    if missing_ratios is None:
        missing_ratios = [0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
    
    # Walk through the output directory
    for city in cities:
        city_path = os.path.join(base_dir, city)
        if not os.path.exists(city_path):
            logger.warning(f"Path does not exist: {city_path}")
            continue
            
        for dirname in os.listdir(city_path):
            dir_path = os.path.join(city_path, dirname)
            
            # Only look for directories with PRAIM model pattern
            if not dirname.startswith('poiradius') or not os.path.isdir(dir_path):
                continue
                
            try:
                # Parse directory name to extract parameters
                # Pattern: poiradius{}_seqlen{}_maskratio{}_*
                parts = dirname.split('_')
                params = {}
                
                for part in parts:
                    if part.startswith('poiradius'):
                        params['poi_radius'] = int(part[9:])
                    elif part.startswith('seqlen'):
                        params['window_size'] = int(part[6:])
                    elif part.startswith('maskratio'):
                        params['missing_ratio'] = float(part[9:])
                
                # Apply filters
                if (params.get('window_size') != window_size or
                    params.get('poi_radius') != poi_radius or
                    params.get('missing_ratio') not in missing_ratios):
                    continue
                
                # Look for experiment directories
                for exp_dirname in os.listdir(dir_path):
                    exp_dir_path = os.path.join(dir_path, exp_dirname)
                    
                    if exp_dirname.startswith('experiment') and os.path.isdir(exp_dir_path):
                        # Check for final_metrics.pkl file
                        result_file = os.path.join(exp_dir_path, "final_metrics.pkl")
                        if os.path.exists(result_file):
                            result_files.append((
                                city, 
                                params.get('window_size'), 
                                params.get('poi_radius'),
                                params.get('missing_ratio'),
                                result_file
                            ))
                            break  # Only take the first experiment directory found
                            
            except Exception as e:
                logger.warning(f"Error parsing directory {dirname}: {e}")
                continue
                
    return result_files



def load_PRAIM_results_to_dataframe(result_files: list) -> pd.DataFrame:
    """
    Load PRAIM results from files into a pandas DataFrame.
    
    Args:
        result_files: List of result file tuples from find_PRAIM_results
        
    Returns:
        DataFrame with all PRAIM results
    """
    records = []
    
    for city, window_size, poi_radius, missing_ratio, file_path in result_files:
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            
            record = {
                'city': city,
                'baseline': 'PRAIM',  # Our model name
                'window_size': window_size,
                'poi_radius': poi_radius,
                'missing_ratio': missing_ratio
            }
            
            # Add metrics (excluding MAPE if present)
            for metric, value in results['imputation'].items():
                if metric != 'imputation_mape':  # Skip MAPE
                    record[metric] = value
            
            records.append(record)
                
        except Exception as e:
            logger.error(f"Error loading PRAIM results from {file_path}: {e}")
    
    return pd.DataFrame(records)




def find_baseline_results(base_dir: str='outputs', cities=None, window_size: int=7, 
                          poi_radius: int=2000, missing_ratios=None) -> list:
    """
    Find all baseline result files based on filtering criteria.
    
    Args:
        base_dir: Base directory for outputs
        cities: List of cities to include (None means all)
        window_size: Window size to filter for (default: 7)
        poi_radius: POI radius to filter for (default: 2000)
        missing_ratios: List of missing ratios to filter for (None means [0.2-0.9])
        
    Returns:
        List of tuples with (city, baseline, window_size, poi_radius, missing_ratio, file_path)
    """
    result_files = []
    
    # Default values if none provided
    if cities is None:
        cities = CITIES
        
    if missing_ratios is None:
        missing_ratios = MISSING_RATIOS # 1, 2, 3, 4, 5, 6 days missing
    
    # Walk through the output directory
    for city in cities:
        city_path = os.path.join(base_dir, city)
        if not os.path.exists(city_path):
            logger.warning(f"Path does not exist: {city_path}")
            continue
            
        for dirname in os.listdir(city_path):
            dir_path = os.path.join(city_path, dirname)
            
            # Skip directories that are not baseline models
            if dirname.startswith('poiradius') or not os.path.isdir(dir_path):
                continue
                
            # Parse directory name to get baseline and parameters
            try:
                # Handle "all" baseline directories differently
                if dirname.startswith("all_"):
                    baseline = "all"
                    parts = dirname[4:].split('_')
                else:
                    baseline = dirname.split('_')[0]
                    # Skip if not a valid baseline
                    if baseline not in VALID_BASELINES and baseline != "all":
                        continue
                    parts = dirname[len(baseline)+1:].split('_')
                
                # Extract parameters
                params = {}
                for part in parts:
                    if part.startswith('poiradius'):
                        params['poi_radius'] = int(part[9:])
                    elif part.startswith('seqlen'):
                        params['window_size'] = int(part[6:])
                    elif part.startswith('maskratio'):
                        params['missing_ratio'] = float(part[9:])
                
                # Apply filters
                if (params.get('window_size', window_size) != window_size or
                    params.get('poi_radius', poi_radius) != poi_radius or
                    params.get('missing_ratio') not in missing_ratios):
                    continue
                
                # Check for results file
                result_file = os.path.join(dir_path, "baseline_results.json")
                if os.path.exists(result_file):
                    result_files.append((
                        city, 
                        baseline, 
                        params.get('window_size', window_size), 
                        params.get('poi_radius', poi_radius),
                        params.get('missing_ratio'),
                        result_file
                    ))
            except Exception as e:
                logger.warning(f"Error parsing directory {dirname}: {e}")
                continue
                
    return result_files



def load_baseline_results_to_dataframe(result_files: list) -> pd.DataFrame:
    """
    Load results from files into a pandas DataFrame.
    
    Args:
        result_files: List of result file tuples from find_baseline_results
        
    Returns:
        DataFrame with all results
    """
    records = []
    
    for city, baseline_dir, window_size, poi_radius, missing_ratio, file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
                
            # If this is from an "all" run, it will have results for multiple baselines
            for baseline, metrics in results.items():
                # Only include valid baseline methods
                if baseline in VALID_BASELINES:
                    record = {
                        'city': city,
                        'baseline': baseline,
                        'window_size': window_size,
                        'poi_radius': poi_radius,
                        'missing_ratio': missing_ratio
                    }
                    # Add metrics (excluding MAPE)
                    for metric, value in metrics.items():
                        if metric != 'imputation_mape':  # Skip MAPE
                            record[metric] = value
                    
                    records.append(record)
                
        except Exception as e:
            logger.error(f"Error loading results from {file_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    
    return df



def visualize_results(df: pd.DataFrame, metrics=None, img_save_dir='imgs'):
    """
    Visualize results with different plots for metrics.
    
    Args:
        df: DataFrame with results
        metrics: List of metrics to visualize (None means all except MAPE)
        output_dir: Directory to save plots
    """
    os.makedirs(img_save_dir, exist_ok=True)
    
    # If no metrics specified, use all available ones
    if metrics is None:
        metrics = METRICS
    elif isinstance(metrics, str):
        metrics = [metrics]  # Convert single string to list
        
    # Map baseline names using BASELINE_MAPPING
    df = df.copy()
    df['baseline_display'] = df['baseline'].map(lambda x: BASELINE_MAPPING.get(x, x))
    
    # Create ordered categories for consistent plotting
    # Filter to only include baselines that exist in the data
    available_baselines = df['baseline'].unique()
    ordered_baselines = [b for b in BASELINE_ORDER if b in available_baselines]
    ordered_display_names = [BASELINE_MAPPING.get(b, b) for b in ordered_baselines]
    
    # Convert to categorical with specified order
    df['baseline'] = pd.Categorical(df['baseline'], categories=ordered_baselines, ordered=True)
    df['baseline_display'] = pd.Categorical(df['baseline_display'], categories=ordered_display_names, ordered=True)
    
    
    # Define mapping from missing ratio to missing days
    ratio_to_days_mapping = {
        0.2: r'$\lambda=0.2$',
        0.3: r'$\lambda=0.3$', 
        0.5: r'$\lambda=0.5$',
        0.6: r'$\lambda=0.6$',
        0.8: r'$\lambda=0.8$',
        0.9: r'$\lambda=0.9$'
    }
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a subdirectory for each metric type
    for metric in metrics:
        os.makedirs(img_save_dir, exist_ok=True)
                
        
        # Compare methods across missing ratios (for each city)
        for city in df['city'].unique():
            if city == 'PaloAlto':
                fig, ax = plt.subplots(figsize=(16, 3.5))
            else:    
                fig, ax = plt.subplots(figsize=(16, 2))
            subset = df[df['city'] == city].sort_values('baseline')
            
            # Sort missing ratios for consistent display
            missing_ratios = sorted(subset['missing_ratio'].unique())
            
            # Create a more appealing color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            if len(missing_ratios) > len(colors):
                colors = plt.cm.viridis(np.linspace(0, 1, len(missing_ratios)))
            else:
                colors = colors[:len(missing_ratios)]
            
            # Create barplot with custom styling
            bars = sns.barplot(
                data=subset, 
                x='baseline_display', 
                y=metric, 
                width=0.9,
                hue='missing_ratio',
                palette=colors,
                order=ordered_display_names,
                ax=ax,
                alpha=0.7,
                edgecolor='white',
                linewidth=3
            )
            
            # Customize the plot
            ax.set_xlabel('Methods', fontsize=22, fontweight='bold')
            ax.set_ylabel('MAE', fontsize=22, fontweight='bold')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='both', labelsize=20)
            
            
            # Make PRAIM label bold
            xticklabels = ax.get_xticklabels()
            for label in xticklabels:
                if label.get_text() == 'PRAIM':
                    label.set_weight('bold')

            
            # Add grid for better readability
            ax.grid(False)
            ax.set_axisbelow(True)
            
            # Add vertical lines to separate method categories with better styling
            if len(ordered_display_names) > 1:
                # Calculate separator positions more accurately
                praim_end = 0.5
                statistical_start = 1 if 'PRAIM' in [BASELINE_MAPPING.get(b, b) for b in ordered_baselines] else 0
                statistical_end = statistical_start + len([b for b in ordered_baselines[1:4] if b in available_baselines]) - 0.5
                ml_end = statistical_end + len([b for b in ordered_baselines[4:8] if b in available_baselines])
                
                # Add separator lines with labels
                if 'PRAIM' in ordered_display_names:
                    ax.axvline(x=praim_end, color='red', linestyle='--', alpha=0.6, linewidth=4)
    
                
                if statistical_end > statistical_start:
                    ax.axvline(x=statistical_end, color='red', linestyle='--', alpha=0.6, linewidth=4)
  
                if ml_end > statistical_end and ml_end < len(ordered_display_names) - 0.5:
                    ax.axvline(x=ml_end, color='red', linestyle='--', alpha=0.6, linewidth=4)
       
            
            
            
            # Customize legend with missing days instead of ratios - ONLY for PaloAlto
            if city == 'PaloAlto':
                handles, labels = ax.get_legend_handles_labels()
                
                # Convert missing ratios to days for legend labels
                legend_labels = []
                for ratio in missing_ratios:
                    if ratio in ratio_to_days_mapping:
                        legend_labels.append(ratio_to_days_mapping[ratio])
                    else:
                        # Fallback for any ratio not in mapping
                        legend_labels.append(f"{float(ratio):.1f}")
                
                legend_props = {
                    'columnspacing': 0.5,
                    'handletextpad': 0.5,
                    'handlelength': 1,
                    'labelspacing': 0.5
                }
                
                legend = ax.legend(
                    handles=handles, 
                    labels=legend_labels,
                    # title='Missing Duration', 
                    bbox_to_anchor=(0.15, 1.6),
                    loc='upper left',
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    ncol=6,
                    fontsize=22,
                    **legend_props
                    # title_fontsize=24
                )
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
                legend.get_frame().set_edgecolor('gray')
            else:
                # Remove legend for other cities
                ax.get_legend().remove()
            
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Add subtle background color
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#fafafa')
            
            # Save with high DPI for better quality
            plt.savefig(
                os.path.join(img_save_dir, f'missing_ratio_comparison_{city}.png'), 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='white',
                edgecolor='none'
            )
            plt.savefig(
                os.path.join(img_save_dir, f'missing_ratio_comparison_{city}.pdf'), 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='white',
                edgecolor='none'
            )
            plt.close()
            
            
              
            
            
            
            

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline imputation methods')
    parser.add_argument('--cities', default=None, 
                        help='Comma-separated list of cities to include')
    parser.add_argument('--window_size', type=int, default=7, 
                        help='Window size to filter for')
    parser.add_argument('--poi_radius', type=int, default=2000, 
                        help='POI radius to filter for')
    parser.add_argument('--img_save_dir', default='imgs', 
                        help='Directory to save results')
    parser.add_argument('--metrics', default='all', 
                        help='Comma-separated metrics to visualize ("all" for all metrics except MAPE)')
    parser.add_argument('--missing_ratios', default='0.2,0.3,0.5,0.6,0.8,0.9', 
                        help='Comma-separated list of missing ratios to evaluate')
    
    args = parser.parse_args()
    
    # Convert cities from comma-separated string to list if provided
    cities = args.cities.split(',') if args.cities else None
    
    # Convert missing_ratios from comma-separated string to list of floats
    missing_ratios = [float(r.strip()) for r in args.missing_ratios.split(',')]
    
    # Process metrics
    if args.metrics.lower() == 'all':
        metrics = METRICS
    else:
        metrics = [m.strip() for m in args.metrics.split(',')]
        # Validate metrics
        invalid_metrics = [m for m in metrics if m not in METRICS]
        if invalid_metrics:
            logger.warning(f"Invalid metrics specified: {invalid_metrics}. Using only valid ones.")
            metrics = [m for m in metrics if m in METRICS]
        
        if not metrics:
            logger.warning("No valid metrics specified. Using 'imputation_rmse' as default.")
            metrics = ['imputation_rmse']
    
    # Find baseline result files
    logger.info("Finding baseline result files...")
    result_files = find_baseline_results(
        cities=cities,
        window_size=args.window_size,
        poi_radius=args.poi_radius,
        missing_ratios=missing_ratios
    )
    
    # Load baseline results into DataFrame
    logger.info("Loading baseline results into dataframe...")
    results_df = load_baseline_results_to_dataframe(result_files)
    

    logger.info("Finding PRAIM result files...")
    praim_files = find_PRAIM_results(
        cities=cities,
        window_size=args.window_size,
        poi_radius=args.poi_radius,
        missing_ratios=missing_ratios
    )
    
    if praim_files:
        logger.info("Loading PRAIM results into dataframe...")
        praim_df = load_PRAIM_results_to_dataframe(praim_files)
        
        # Combine baseline and PRAIM results
        results_df = pd.concat([results_df, praim_df], ignore_index=True)
        logger.info(f"Found {len(praim_files)} PRAIM result files.")
    else:
        logger.warning("No PRAIM result files found with the specified criteria.")
    
    if len(results_df) == 0:
        logger.warning("No result files found with the specified criteria.")
        return
        
    logger.info(f"Found {len(result_files)} baseline result files.")
    
    # Save raw data
    os.makedirs(args.img_save_dir, exist_ok=True)

    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualize_results(results_df, metrics=metrics, img_save_dir=args.img_save_dir)
    
    logger.info(f"Evaluation complete. Results saved to {args.img_save_dir}")
    
    
    

if __name__ == "__main__":
    main()