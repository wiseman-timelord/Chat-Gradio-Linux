# launcher.py

import os
import time
from pathlib import Path
import subprocess
from datetime import datetime
import scripts.temporary as temporary
from scripts.settings import load_config
from scripts.interface import launch_interface

print("Starting launcher"); time.sleep(1)

def check_cuda_availability():
    """Verify CUDA is available and unified memory is supported."""
    try:
        # Check for nvidia-smi
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        if "NVIDIA-SMI" not in nvidia_smi:
            print("NVIDIA driver not loaded"); time.sleep(3)
            return False

        # Check for CUDA toolkit
        try:
            cuda_version = subprocess.check_output("nvcc --version", shell=True).decode()
            if "release" not in cuda_version:
                print("CUDA toolkit missing"); time.sleep(3)
                return False
        except:
            print("CUDA toolkit not found"); time.sleep(3)
            return False

        # Check binary supports unified memory
        binary_path = Path("data/llama-cpp/main")
        if not binary_path.exists():
            print("Binary missing"); time.sleep(3)
            return False

        help_output = subprocess.check_output(f"{binary_path} --help", shell=True).decode()
        if "unified" not in help_output.lower():
            print("Unified memory not enabled in binary"); time.sleep(3)
            return False

        return True
    except subprocess.CalledProcessError:
        print("CUDA unavailable"); time.sleep(3)
        return False
    except Exception as e:
        print(f"CUDA check error: {str(e)}"); time.sleep(3)
        return False

def main():
    try:
        print("Launcher initializing"); time.sleep(1)
        script_dir = Path(__file__).parent.resolve()
        os.chdir(script_dir)
        print("Directory set"); time.sleep(1)
        temporary.DATA_DIR = str(script_dir / "data")
        print("Data directory set"); time.sleep(1)
        Path(temporary.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        
        # Pre-flight checks
        binary_path = Path("data/llama-cpp/main")
        if not binary_path.exists():
            print("Binary missing"); time.sleep(3)
            raise FileNotFoundError("llama.cpp binary not found")
        
        if not check_cuda_availability():
            print("CUDA not detected"); time.sleep(3)
            raise RuntimeError("CUDA unavailable")
        
        print("Loading config"); time.sleep(1)
        config_status = load_config()
        print("Config loaded"); time.sleep(1)
        
        print("Launching interface"); time.sleep(1)
        try:
            launch_interface()
        except Exception as e:
            print("Interface launch failed"); time.sleep(3)
            raise
    except Exception as e:
        print("Launcher error"); time.sleep(3)
        raise

if __name__ == "__main__":
    main()