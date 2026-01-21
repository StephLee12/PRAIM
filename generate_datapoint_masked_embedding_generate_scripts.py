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
    
    
    # Loop through each combination of city and missing ratio
    for city in cities:

        script_name = f"generate_datapoint_masked_embedding_{city}.sh"

        
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


echo "Starting data processing with different missing ratios..."

# Loop through missing ratios from 0.0 to 0.9 with increments of 0.1
for ratio in $(seq 0.0 0.1 0.9); do
  echo "====================================="
  echo "Running with missing_ratio = $ratio"
  echo "====================================="
  
  python generate_datapoint_masked_embedding.py --missing_ratio $ratio --city {city}
  
  echo "Completed missing_ratio = $ratio"
  echo "-------------------------------------"
done

echo "All processing complete!"
"""
            
        # Write the script to a file
        with open(script_name, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_name, 0o755)
        
        print(f"Generated: {script_name}")

if __name__ == "__main__":
    generate_baseline_scripts()