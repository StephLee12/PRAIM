#!/usr/bin/env python3
# filepath: generate_baseline_scripts.py

import os

def generate_baseline_scripts():
    """
    Generate bash scripts for running baseline experiments with different 
    cities and missing ratios.
    """
    # Define parameters
    # cities = ["PaloAlto", "Boulder", "Dundee", "Perth"]
    cities = ["Dundee", "Perth"]
    missing_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    
    # Loop through each combination of city and missing ratio
    for city in cities:
        for ratio in missing_ratios:
            ratio_str = f"{ratio:.1f}"
            script_name = f"train_decoder_{city}_maskratio{ratio_str}.sh"

            
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


# python train_decoder.py --city {city} --missing_ratio {ratio_str}
python train_decoder_cp.py --city {city} --missing_ratio {ratio_str}

echo "City:{city}; Missing Ratio:{ratio_str} Completed!"
"""
            
            # Write the script to a file
            with open(script_name, 'w') as f:
                f.write(script_content)
            
            # Make the script executable
            os.chmod(script_name, 0o755)
            
            print(f"Generated: {script_name}")

if __name__ == "__main__":
    generate_baseline_scripts()