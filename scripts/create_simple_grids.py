#!/usr/bin/env python3
"""
Simplified MLflow Experiment Grid Creator
"""

import mlflow
import mlflow.tracking
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

def setup_mlflow_tracking():
    """Setup MLflow tracking URI."""
    current_dir = Path.cwd()
    mlruns_path = None
    
    for path in [current_dir] + list(current_dir.parents):
        potential_mlruns = path / "mlruns"
        if potential_mlruns.exists():
            mlruns_path = potential_mlruns
            break
    
    if mlruns_path:
        tracking_uri = f"file://{mlruns_path.absolute()}"
    else:
        tracking_uri = "file://./mlruns"
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

def get_clustering_experiments():
    """Get clustering experiments with images."""
    client = mlflow.tracking.MlflowClient()
    experiments = []
    
    # Get all experiments
    all_experiments = client.search_experiments()
    
    for experiment in all_experiments:
        exp_name = experiment.name.lower()
        
        # Filter clustering experiments
        if any(keyword in exp_name for keyword in ["kmeans", "cluster"]):
            print(f"Processing experiment: {experiment.name}")
            
            # Get runs for this experiment
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=20
            )
            
            for run in runs:
                # Check for artifacts (images)
                try:
                    artifacts = client.list_artifacts(run.info.run_id)
                    has_image = any(art.path.lower().endswith('.png') for art in artifacts)
                    
                    if has_image:
                        # Extract metadata
                        run_info = {
                            'run_id': run.info.run_id,
                            'run_name': run.info.run_name,
                            'experiment_name': experiment.name,
                            'metrics': dict(run.data.metrics),
                            'params': dict(run.data.params),
                        }
                        
                        # Extract layer
                        layer = None
                        layer_param = run_info['params'].get('layer_num') or run_info['params'].get('layer')
                        if layer_param:
                            layer = int(layer_param)
                        else:
                            layer_match = re.search(r'layer(\d+)', run_info['run_name'])
                            if layer_match:
                                layer = int(layer_match.group(1))
                        
                        run_info['layer'] = layer
                        
                        # Extract purity and NMI
                        purity = None
                        nmi = None
                        for metric_name, value in run_info['metrics'].items():
                            if 'purity' in metric_name.lower():
                                purity = value
                            elif 'nmi' in metric_name.lower():
                                nmi = value
                        
                        run_info['purity'] = purity
                        run_info['nmi'] = nmi
                        
                        # Check if it's EC
                        is_ec = 'ec' in run_info['run_name'].lower() or 'ec' in experiment.name.lower()
                        run_info['is_ec'] = is_ec
                        
                        # Determine dataset
                        dataset = 'snli'
                        if 'folio' in experiment.name.lower() or 'folio' in run_info['run_name'].lower():
                            dataset = 'folio'
                        run_info['dataset'] = dataset
                        
                        experiments.append(run_info)
                        
                except Exception as e:
                    print(f"Error processing run {run.info.run_id}: {e}")
    
    return experiments

def get_probing_experiments():
    """Get probing experiments with images."""
    client = mlflow.tracking.MlflowClient()
    experiments = []
    
    # Get all experiments
    all_experiments = client.search_experiments()
    
    for experiment in all_experiments:
        exp_name = experiment.name.lower()
        
        # Filter probing experiments
        if any(keyword in exp_name for keyword in ["probe", "probing", "decision"]):
            print(f"Processing experiment: {experiment.name}")
            
            # Get runs for this experiment
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=20
            )
            
            for run in runs:
                # Check for artifacts (images)
                try:
                    artifacts = client.list_artifacts(run.info.run_id)
                    has_image = any(art.path.lower().endswith('.png') for art in artifacts)
                    
                    if has_image:
                        # Extract metadata
                        run_info = {
                            'run_id': run.info.run_id,
                            'run_name': run.info.run_name,
                            'experiment_name': experiment.name,
                            'metrics': dict(run.data.metrics),
                            'params': dict(run.data.params),
                        }
                        
                        # Extract layer
                        layer = None
                        layer_param = run_info['params'].get('layer_num') or run_info['params'].get('layer')
                        if layer_param:
                            layer = int(layer_param)
                        else:
                            layer_match = re.search(r'layer(\d+)', run_info['run_name'])
                            if layer_match:
                                layer = int(layer_match.group(1))
                        
                        run_info['layer'] = layer
                        
                        # Extract accuracy
                        accuracy = run_info['metrics'].get('accuracy')
                        run_info['accuracy'] = accuracy
                        
                        # Check if it's EC
                        is_ec = 'ec' in run_info['run_name'].lower() or 'ec' in experiment.name.lower()
                        run_info['is_ec'] = is_ec
                        
                        # Determine dataset
                        dataset = 'snli'
                        if 'folio' in experiment.name.lower() or 'folio' in run_info['run_name'].lower():
                            dataset = 'folio'
                        run_info['dataset'] = dataset
                        
                        experiments.append(run_info)
                        
                except Exception as e:
                    print(f"Error processing run {run.info.run_id}: {e}")
    
    return experiments

def get_image_for_run(run_id):
    """Download image for a run."""
    client = mlflow.tracking.MlflowClient()
    
    try:
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            if artifact.path.lower().endswith('.png'):
                local_path = client.download_artifacts(run_id, artifact.path)
                return local_path
    except Exception as e:
        print(f"Error getting image for {run_id}: {e}")
    
    return None

def create_grid(experiments, experiment_type, dataset, output_path):
    """Create a 3x3 grid of experiment images."""
    import random
    
    # Filter by dataset and prioritize EC
    filtered = [exp for exp in experiments if exp['dataset'] == dataset]
    
    # Sort: EC first, then by metric
    if experiment_type == "clustering":
        filtered.sort(key=lambda x: (not x['is_ec'], -(x['purity'] or 0)))
    else:  # probing
        filtered.sort(key=lambda x: (not x['is_ec'], -(x['accuracy'] or 0)))
    
    # Random selection from all experiments
    if len(filtered) >= 9:
        selected = random.sample(filtered, 9)
    else:
        selected = filtered
    
    if not selected:
        print(f"No {experiment_type} experiments found for {dataset}")
        return False
    
    print(f"Creating 3x3 grid for {len(selected)} {experiment_type} experiments ({dataset}) - random sample")
    
    # Create 3x3 grid with higher resolution
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'{experiment_type.title()} Experiments - {dataset.upper()}\n(Muestra aleatoria de configuraciones)', 
                 fontsize=18, fontweight='bold')
    
    for idx, exp in enumerate(selected):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Get image
        image_path = get_image_for_run(exp['run_id'])
        
        if image_path and os.path.exists(image_path):
            try:
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.axis('off')
                
                # Add metadata text
                layer = exp['layer'] or 'N/A'
                
                if experiment_type == "clustering":
                    purity = f"{exp['purity']:.4f}" if exp['purity'] else 'N/A'
                    nmi = f"{exp['nmi']:.6f}" if exp['nmi'] else 'N/A'
                    text = f"Layer: {layer}\nPurity: {purity}\nNMI: {nmi}"
                else:  # probing
                    accuracy = f"{exp['accuracy']:.4f}" if exp['accuracy'] else 'N/A'
                    text = f"Layer: {layer}\nAccuracy: {accuracy}"
                
                ax.text(0.5, -0.12, text, transform=ax.transAxes, 
                       ha='center', va='top', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgray', alpha=0.8))
                
                # Add EC indicator
                if exp['is_ec']:
                    ax.text(0.05, 0.95, 'EC', transform=ax.transAxes,
                           ha='left', va='top', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.9),
                           color='white')
                
            except Exception as e:
                print(f"Error displaying image: {e}")
                ax.text(0.5, 0.5, 'Image\nError', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No Image', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(selected), 9):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Grid saved to: {output_path}")
    return True

def main():
    setup_mlflow_tracking()
    
    # Create output directory
    output_dir = Path("reports_src/images/grids")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Getting clustering experiments...")
    clustering_experiments = get_clustering_experiments()
    print(f"Found {len(clustering_experiments)} clustering experiments with images")
    
    print("Getting probing experiments...")
    probing_experiments = get_probing_experiments()
    print(f"Found {len(probing_experiments)} probing experiments with images")
    
    # Create grids
    print("\nCreating grids...")
    
    # SNLI clustering
    create_grid(clustering_experiments, "clustering", "snli", 
                output_dir / "snli_clustering_grid.png")
    
    # FOLIO clustering
    create_grid(clustering_experiments, "clustering", "folio", 
                output_dir / "folio_clustering_grid.png")
    
    # SNLI probing
    create_grid(probing_experiments, "probing", "snli", 
                output_dir / "snli_probing_grid.png")
    
    # FOLIO probing
    create_grid(probing_experiments, "probing", "folio", 
                output_dir / "folio_probing_grid.png")
    
    print(f"\nAll grids saved to: {output_dir}")

if __name__ == "__main__":
    main() 