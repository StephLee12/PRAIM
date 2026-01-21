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

        script_name = f"generate_datapoint_embedding_{city}.sh"

        
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


python generate_datapoint_embedding.py --city {city}

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