import os
import yaml
import shutil
import subprocess
import time
import argparse

YAML_DIR = "scratch/tmp_yamls/linear"

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

def run_merge(yaml_path, output_dir):
    """Run mergekit-yaml command."""
    cmd = ["mergekit-yaml", yaml_path, output_dir, "--cuda", "--trust-remote-code"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_push_to_hub(output_dir):
    """Run push_to_hub.py command."""
    cmd = ["python", "push_to_hub_linear.py", "--path", output_dir]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run model merges with varying weights.')
    parser.add_argument('--weight-start', type=float, default=0.5,
                        help='Starting weight for the first model (default: 0.5)')
    parser.add_argument('--weight-end', type=float, default=0.5,
                        help='Ending weight for the first model (default: 0.5)')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Step size between weights (default: 0.1)')
    parser.add_argument('--recipe', type=str, default=None,
                        help=f'Path to the recipe YAML file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'Base output directory')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load the original YAML file
    original_yaml = load_yaml(args.recipe)
    
    # Create a temp directory for modified YAML files
    os.makedirs(YAML_DIR, exist_ok=True)
    
    # Calculate weights based on step size
    weights = []
    current = args.weight_start
    while current <= args.weight_end + 1e-6 if args.weight_start <= args.weight_end else current >= args.weight_end - 1e-6:
        weights.append(round(current, 3))
        if args.weight_start == args.weight_end:  # Only one weight
            break
        current = round(current + args.step_size if args.weight_start < args.weight_end else current - args.step_size, 3)
    
    # Process each weight
    for weight_1 in weights:
        weight_2 = round(1.0 - weight_1, 3)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING WEIGHTS: {weight_1} and {weight_2}")
        print(f"{'='*60}\n")
        
        # Create output directory for this specific weight
        current_output_dir = f"{args.output_dir}_{weight_1}_{weight_2}"
        
        # Create modified YAML file
        modified_yaml = update_weights(original_yaml.copy(), weight_1, weight_2)
        temp_yaml_path = f"{YAML_DIR}/weights_{weight_1}_{weight_2}.yml"
        save_yaml(modified_yaml, temp_yaml_path)
        
        # Ensure output directory is clean
        if os.path.exists(current_output_dir):
            shutil.rmtree(current_output_dir)
        
        # Run merge
        try:
            run_merge(temp_yaml_path, current_output_dir)
            
            # Push to hub
            run_push_to_hub(current_output_dir)
            
            print(f"Successfully processed weights {weight_1} and {weight_2}")
        except Exception as e:
            print(f"Error processing weights {weight_1} and {weight_2}: {e}")
            
        # Add a short pause between iterations
        time.sleep(2)

if __name__ == "__main__":
    main()