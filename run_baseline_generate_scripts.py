#!/usr/bin/env python3
# filepath: generate_baseline_scripts.py

import os

def generate_baseline_scripts():
    """
    Generate bash scripts for running baseline experiments with different 
    cities and missing ratios.
    """
    # Define parameters
    cities = ["PaloAlto", "Boulder", "Dundee", "Perth"]
    missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Baselines from the original script
    baselines = [
        "mean", "zero", "last_observed", "interpolation", 
        "knn", "svd", "kalman", "missforest", "lstm", 
        "transformer", "tcn", "gan", "autoencoder"
    ]
    
    # Loop through each combination of city and missing ratio
    for city in cities:
        for ratio in missing_ratios:
            ratio_str = f"{ratio:.1f}"
            script_name = f"run_baseline_{city}_maskratio{ratio_str}.sh"
            
            # Create bash array of baselines
            baselines_array = " ".join([f'"{b}"' for b in baselines])
            
            # Generate script content
            script_content = f"""#!/bin/bash

#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=50GB
#PBS -l jobfs=50GB
#PBS -q gpuvolta 
#PBS -P nq33
#PBS -l walltime=24:00:00
#PBS -l storage=scratch/nq33
#PBS -l wd


source ~/.bashrc
source /scratch/nq33/jl7986/pipenv_hyperPred/bin/activate
module load cuda/12.5.1


# Array of baseline methods
BASELINES=({baselines_array})

# Loop through each baseline and run the script
for baseline in "${{BASELINES[@]}}"; do
    echo "Running baseline: $baseline"
    python run_baseline.py --city {city} --baseline "$baseline" --missing_ratio {ratio_str}
    echo "Completed baseline: $baseline"
    echo "-------------------------------------"
done

echo "All baselines completed!"
"""
            
            # Write the script to a file
            with open(script_name, 'w') as f:
                f.write(script_content)
            
            # Make the script executable
            os.chmod(script_name, 0o755)
            
            print(f"Generated: {script_name}")

if __name__ == "__main__":
    generate_baseline_scripts()