import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse

# Set style for better appearance
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'legend.framealpha': 0.9
})

def load_evaluation_results(results_dir: str, cities: List[str]) -> Dict[str, Dict]:
    """
    Load evaluation results for all cities.
    
    Args:
        results_dir: Directory containing evaluation results
        cities: List of cities to load
        
    Returns:
        Dictionary with results for each city
    """
    all_results = {}
    
    for city in cities:
        results_file = os.path.join(results_dir, f"{city}_downstream_forecast_evaluation.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results[city] = json.load(f)
            print(f"Loaded results for {city}")
        else:
            print(f"Warning: Results file not found for {city}: {results_file}")
    
    return all_results

def extract_relative_improvements(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Extract relative improvements from evaluation results.
    
    Args:
        results: Raw evaluation results
        
    Returns:
        Processed relative improvements data
    """
    improvements_data = {}
    
    for city, city_results in results.items():
        if 'relative_improvements' not in city_results:
            print(f"No relative improvements found for {city}")
            continue
            
        improvements_data[city] = city_results['relative_improvements']
    
    return improvements_data

def plot_relative_performance(improvements_data: Dict[str, Dict], output_dir: str):
    """
    Plot relative performance for each city showing MAE and MSE in a single plot.
    
    Args:
        improvements_data: Relative improvements data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot
    metrics = ['MAE', 'MSE']
    
    # Define beautiful colors for each metric with better contrast
    metric_colors = {
        'MAE': '#2E86AB',  # Deep blue
        'MSE': '#F24236'   # Coral red
    }
    
    # Get all models from the first available city
    all_models = set()
    for city_data in improvements_data.values():
        for method_data in city_data.values():
            all_models.update(method_data.keys())
    all_models = sorted(list(all_models))
    
    # Create a plot for each city
    for city, city_data in improvements_data.items():
        # Get imputation methods for this city
        methods = list(city_data.keys())
        
        if not methods:
            print(f"No methods found for {city}")
            continue
        
        # Create single plot with better figure size
        fig, ax = plt.subplots(1, 1, figsize=(7, 2))
        
        # Set background color
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFAFA')
        
        # Prepare combined data for both metrics
        n_methods = len(methods)
        n_models = len(all_models)
        n_metrics = len(metrics)
        
        # Create x-axis positions for models
        model_positions = np.arange(n_models)
        
        # Width of bars
        bar_width = 0.8 / (n_methods * n_metrics)
        
        # Track which metrics have been added to legend
        legend_added = set()
        
        # Calculate the center position for each model's group of bars
        all_bar_positions = []
        
        # Plot bars for each method and metric combination
        for method_idx, method in enumerate(methods):
            for metric_idx, metric in enumerate(metrics):
                method_metric_values = []
                
                for model in all_models:
                    if (model in city_data[method] and 
                        metric in city_data[method][model]):
                        improvement = city_data[method][model][metric]
                    else:
                        improvement = 0
                    method_metric_values.append(improvement)
                
                # Calculate positions for this method-metric combination
                offset = (method_idx * n_metrics + metric_idx - (n_methods * n_metrics - 1) / 2) * bar_width
                positions = model_positions + offset
                all_bar_positions.extend(positions)
                
                # Create label for legend (only once per metric)
                label = metric if metric not in legend_added else None
                if label:
                    legend_added.add(metric)
                
                # Create bars with metric-specific colors and edge lines
                bars = ax.bar(positions, method_metric_values, bar_width, 
                             label=label, alpha=0.8, color=metric_colors[metric],
                             edgecolor='white', linewidth=0.8)
                
                # Add subtle gradient effect by varying alpha slightly
                for i, bar in enumerate(bars):
                    if method_metric_values[i] > 0:
                        bar.set_alpha(0.9)
                    elif method_metric_values[i] < 0:
                        bar.set_alpha(0.7)
                    else:
                        bar.set_alpha(0.5)
        
        # Calculate the center position for each model's group of bars
        tick_positions = []
        for model_idx in range(n_models):
            # Find all bar positions for this model
            model_bar_positions = []
            for method_idx, method in enumerate(methods):
                for metric_idx, metric in enumerate(metrics):
                    offset = (method_idx * n_metrics + metric_idx - (n_methods * n_metrics - 1) / 2) * bar_width
                    position = model_positions[model_idx] + offset
                    model_bar_positions.append(position)
            
            # Calculate center of this model's bars
            center_position = (min(model_bar_positions) + max(model_bar_positions)) / 2
            tick_positions.append(center_position)
        
        # Customize the plot with better styling
        ax.set_xlabel('Forecasting Models', fontweight='bold', fontsize=14)
        ax.set_ylabel('Imp. (%)', fontweight='bold', fontsize=14)
        ax.set_xticks(tick_positions)  # Use calculated center positions
        ax.set_xticklabels(all_models, ha='center', fontsize=11)  # Center alignment
        
        # Style the legend
        legend = ax.legend(loc='upper left', fontsize=13, framealpha=0.95)
        legend.get_frame().set_edgecolor('#BDC3C7')
        legend.get_frame().set_linewidth(1.2)
        
        # Enhanced grid
        # ax.grid(True, alpha=0.4, axis='y', linestyle='--', linewidth=0.8)
        ax.grid(False)
        ax.set_axisbelow(True)
        
        # Add horizontal line at y=0 with better styling
        ax.axhline(y=0, color='#7F8C8D', linestyle='-', alpha=0.8, linewidth=1.5)
        
        # Set y-axis limits for better visibility
        y_values = []
        for method_data in city_data.values():
            for model_data in method_data.values():
                for metric in metrics:
                    if metric in model_data:
                        y_values.append(model_data[metric])
        
        if y_values:
            y_min, y_max = min(y_values), max(y_values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)
        
        # Style the axes
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#BDC3C7')
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        
        # Adjust layout with better spacing
        plt.tight_layout(pad=1.5)
        
        # Save plot with higher quality
        plot_file = os.path.join(output_dir, f"{city}_relative_performance.png")
        plt.savefig(plot_file, dpi=400, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.savefig(plot_file.replace('.png', '.pdf'), bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        print(f"Saved plot for {city}: {plot_file}")
        
        plt.close()


def main():
    """Main function to create all plots."""
    parser = argparse.ArgumentParser(description="Plot relative performance of imputation methods on downstream forecasting")
    
    parser.add_argument('--results_dir', type=str, default='res_downstream_forecast',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default='plots_downstream_forecast',
                       help='Directory to save plots')
    parser.add_argument('--cities', type=str, nargs='+', 
                       default=['PaloAlto', 'Boulder', 'Dundee', 'Perth'],
                       help='Cities to plot')
    
    args = parser.parse_args()
    
    print("Loading evaluation results...")
    results = load_evaluation_results(args.results_dir, args.cities)
    
    if not results:
        print("No results found. Please check the results directory and city names.")
        return
    
    print("Extracting relative improvements...")
    improvements_data = extract_relative_improvements(results)
    
    if not improvements_data:
        print("No relative improvements data found. Please check the evaluation results.")
        return
    
    print("Creating individual city plots...")
    plot_relative_performance(improvements_data, args.output_dir)
    
    print(f"All plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()