import os
import yaml
import shutil
import subprocess
import argparse
import re
from huggingface_hub import list_repo_refs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a model soup from multiple revisions.")
    parser.add_argument("--model_id", type=str, required=True, help="The Hugging Face model ID.")
    parser.add_argument("--revision", type=str, required=True, help="Revision to filter (e.g., v03.00).")
    parser.add_argument("--output_dir", type=str, default="scratch/model_soup", help="Output directory for the merged model.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type for merged model.")
    parser.add_argument("--chat_template", type=str, default="auto", help="Chat template to use.")
    return parser.parse_args()

def load_yaml(file_path):
    """Load YAML file and return its content."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def save_yaml(content, file_path):
    """Save content to YAML file."""
    with open(file_path, "w") as file:
        yaml.dump(content, file, default_flow_style=False, sort_keys=False)

def get_model_revisions(model_id, revision_pattern):
    """Get all revisions of a model matching the revision pattern."""
    pattern = re.compile(f"{revision_pattern}-step-\\d+$")
    
    try:
        refs = list_repo_refs(model_id)
        
        matching_revisions = []
        for ref in refs.branches:
            if pattern.search(ref.name):
                matching_revisions.append(ref.name)
        
        matching_revisions.sort(key=lambda rev: int(re.search(r"step-(\d+)$", rev).group(1)))
        
        return matching_revisions
    except Exception as e:
        print(f"Error fetching model revisions: {e}")
        return []

def create_merge_recipe(model_id, revisions, output_path, dtype="bfloat16", chat_template="auto"):
    """Create a YAML recipe for linear merging."""
    models = []
    
    weight = 1.0 / len(revisions)
    weights = [weight] * len(revisions)
    
    for revision, weight in zip(revisions, weights):
        models.append({
            "model": f"{model_id}@{revision}",
            "parameters": {
                "weight": weight
            }
        })
    
    recipe = {
        "models": models,
        "merge_method": "linear",
        "dtype": dtype,
        "chat_template": chat_template
    }
    
    save_yaml(recipe, output_path)
    return output_path

def run_merge(yaml_path, output_dir):
    """Run mergekit-yaml command."""
    cmd = ["mergekit-yaml", yaml_path, output_dir, "--cuda"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_push_to_hub(output_dir, revision):
    """Run push_to_hub.py command."""
    cmd = ["python", "push_to_hub_linear.py", "--path", output_dir, "--revision", f"{revision}_soup"]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"MODEL SOUP CREATION: {args.model_id} - {args.revision}")
    print(f"{'='*60}\n")
    
    # Get matching revisions
    print(f"Finding revisions matching {args.revision}-step-XX...")
    revisions = get_model_revisions(args.model_id, args.revision)
    
    if not revisions:
        print(f"No matching revisions found for {args.model_id} with revision pattern {args.revision}")
        return
    
    print(f"Found {len(revisions)} matching revisions:")
    for rev in revisions:
        print(f"  - {rev}")
    
    # Create directory for temporary files
    os.makedirs("scratch/temp_yamls", exist_ok=True)
    
    # Create merge recipe
    recipe_path = f"scratch/temp_yamls/{args.model_id.split('/')[-1]}_{args.revision}_soup.yml"
    create_merge_recipe(
        args.model_id, 
        revisions, 
        recipe_path, 
        dtype=args.dtype,
        chat_template=args.chat_template
    )
    
    print(f"Created merge recipe at {recipe_path}")
    
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    try:
        run_merge(recipe_path, args.output_dir)
        run_push_to_hub(args.output_dir, args.revision)        
        print(f"Successfully created model soup for {args.model_id} - {args.revision}")
    except Exception as e:
        print(f"Error creating model soup: {e}")

if __name__ == "__main__":
    main()