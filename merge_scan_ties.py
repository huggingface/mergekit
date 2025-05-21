import os
import yaml
import shutil
import subprocess
import time
import numpy as np
import argparse

# Define paths
RECIPE_PATH = "recipes/R1-Distill-Qwen-Math-7B/v00.02_v01.02_ties.yml"
OUTPUT_DIR = "scratch/v00.02_v01.02_ties"

def load_yaml(file_path):
    """Load YAML file and return its content."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def save_yaml(content, file_path):
    """Save content to YAML file."""
    with open(file_path, "w") as file:
        yaml.dump(content, file, default_flow_style=False, sort_keys=False)

def update_weights(yaml_content, weight_1, weight_2):
    """Update weights in the YAML content."""
    yaml_content["models"][0]["parameters"]["weight"] = weight_1
    yaml_content["models"][1]["parameters"]["weight"] = weight_2
    return yaml_content

def update_lambda(yaml_content, new_lambda):
    """Update lambda in the YAML content."""
    yaml_content["parameters"]["lambda"] = new_lambda
    return yaml_content

def update_densities(yaml_content, density_1, density_2):
    """Update density values in the YAML content."""
    yaml_content["models"][0]["parameters"]["density"] = density_1
    yaml_content["models"][1]["parameters"]["density"] = density_2
    return yaml_content

def run_merge(yaml_path, output_dir):
    """Run mergekit-yaml command."""
    cmd = ["mergekit-yaml", yaml_path, output_dir, "--cuda"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_push_to_hub(output_dir):
    """Run push_to_hub.py command."""
    cmd = ["python", "push_to_hub_ties.py", "--path", output_dir]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run model merges with different lambda and density values")
    parser.add_argument("--lambda-start", type=float, default=1.0,
                        help="Starting lambda value (default: 1.0)")
    parser.add_argument("--lambda-end", type=float, default=None,
                        help="Ending lambda value (if specified, creates a range; default: None)")
    parser.add_argument("--lambda-step", type=float, default=0.1,
                        help="Step size for lambda range (default: 0.1)")
    parser.add_argument("--density-start", type=float, default=0.2,
                        help="Starting density value (default: 0.2)")
    parser.add_argument("--density-end", type=float, default=None,
                        help="Ending density value for 2D density scan (default: None)")
    parser.add_argument("--density-step", type=float, default=0.1,
                        help="Step size for density range (default: 0.1)")
    parser.add_argument("--recipe", type=str, default=RECIPE_PATH,
                        help=f"Path to the recipe YAML file (default: {RECIPE_PATH})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory for merged model (default: {OUTPUT_DIR})")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load the original YAML file
    original_yaml = load_yaml(args.recipe)
    
    # Create a temp directory for modified YAML files
    os.makedirs("scratch/temp_yamls_ties", exist_ok=True)
    
    # Determine lambda values to process
    if args.lambda_end is None:
        # Single lambda value
        lambda_values = [args.lambda_start]
    else:
        # Range of lambda values
        lambda_values = [round(x, 2) for x in np.arange(
            args.lambda_start, 
            args.lambda_end + args.lambda_step/2,  # Add half step to ensure end value is included
            args.lambda_step
        )]
    
    # Determine density pairs to process
    if args.density_start is None or args.density_end is None:
        # No density scan, use None to indicate no density update needed
        density_pairs = [(None, None)]
    else:
        # Generate density pairs with standard Python float values
        density_pairs = []
        step = float(args.density_step)
        start = float(args.density_start)
        end = float(args.density_end) + step/2  # Include the end value with a small buffer

        # Create ranges with numpy but convert to regular floats and round them
        for d1 in np.arange(start, end, step):
            for d2 in np.arange(start, end, step):
                # Convert numpy floats to Python native floats and round to 2 decimal places
                density_pairs.append((round(float(d1), 2), round(float(d2), 2)))
    
    # Scan through lambda values and density pairs
    for lmda in lambda_values:
        for density_pair in density_pairs:
            density_1, density_2 = density_pair
            
            # Create descriptive output strings
            lambda_str = f"lambda_{lmda}"
            density_str = "" if density_1 is None else f"_density_{density_1}_{density_2}"
            
            print(f"\n{'='*60}")
            print(f"PROCESSING LAMBDA: {lmda}{' WITH DENSITIES: ' + str(density_pair) if density_1 is not None else ''}")
            print(f"{'='*60}\n")
            
            # Create modified YAML file
            modified_yaml = update_lambda(original_yaml.copy(), lmda)
            
            # Update densities if specified
            if density_1 is not None:
                modified_yaml = update_densities(modified_yaml, density_1, density_2)
            
            temp_yaml_path = f"scratch/temp_yamls_ties/v00.00_v01.00_{lambda_str}{density_str}.yml"
            save_yaml(modified_yaml, temp_yaml_path)
            
            # Ensure output directory is clean
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)
            
            # Run merge
            try:
                run_merge(temp_yaml_path, args.output_dir)
                
                # Push to hub
                run_push_to_hub(args.output_dir)
                
                print(f"Successfully processed {lambda_str}{density_str}")
            except Exception as e:
                print(f"Error processing {lambda_str}{density_str}: {e}")
                
            # Add a short pause between iterations
            time.sleep(2)

if __name__ == "__main__":
    main()