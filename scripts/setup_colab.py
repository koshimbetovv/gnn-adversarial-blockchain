import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True)

def main():
    print("Detected environment:")
    try:
        import torch
        print(f"  Torch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            dev_str = "cu" + torch.version.cuda.replace(".", "")
        else:
            dev_str = "cpu"
    except ImportError:
        print("  Torch not installed yet. Installing default...")
        run_command("pip install torch")
        import torch
        dev_str = "cpu"

    # Install basic requirements
    print("\nInstalling requirements...")
    run_command("pip install -r requirements.txt")

    # Install PyG dependencies with correct wheels
    # Colab usually has torch pre-installed. We need to match the version.
    torch_ver = torch.__version__.split("+")[0]  # e.g. 2.5.1
    
    # map common colab cuda versions to wheel strings if needed, 
    # but usually PyG provides wheels for standard torch versions.
    # We will try to use the pip install with -f
    
    print(f"\nInstalling PyG dependencies for torch={torch_ver}, device={dev_str}...")
    
    # Official PyG one-liner doesn't always work for all sub-dependencies on Colab without -f
    # URL pattern: https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
    
    # Adjust dev_str for wheel url
    # e.g. torch-2.5.0+cu121 -> https://data.pyg.org/whl/torch-2.5.0+cu121.html
    # but torch.version.cuda is like '12.1'.
    
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda  # '12.1'
        cuda_suffix = "cu" + cuda_ver.replace('.', '') # 'cu121'
    else:
        cuda_suffix = 'cpu'
        
    whl_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_suffix}.html"
    
    try:
        cmds = [
            f"pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f {whl_url}",
            "pip install torch-geometric"
        ]
        for cmd in cmds:
            run_command(cmd)
            
    except Exception as e:
        print(f"\nError installing PyG dependencies: {e}")
        print("You may need to manually install them based on your specific Colab runtime.")

    print("\nSetup complete!")

if __name__ == "__main__":
    main()
