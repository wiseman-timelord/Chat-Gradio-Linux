# launcher.py

import os, re, time
from pathlib import Path
import subprocess
import scripts.temporary as temporary
from scripts.settings import load_config
from scripts.interface import launch_interface

print("Starting launcher"); time.sleep(1)

def check_cuda_availability():
    """Verify CUDA is available and unified memory is supported."""
    try:
        # Check NVIDIA driver
        try:
            nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
            if "NVIDIA-SMI" not in nvidia_smi:
                print("NVIDIA driver not loaded"); time.sleep(1)
                return False
        except:
            print("NVIDIA driver not found"); time.sleep(1)
            return False

        # Check CUDA toolkit
        try:
            cuda_version = subprocess.check_output("nvcc --version", shell=True).decode()
            if "release" not in cuda_version:
                print("CUDA toolkit missing"); time.sleep(1)
                return False
        except:
            print("CUDA toolkit not found"); time.sleep(1)
            return False

        # Check VRAM
        try:
            smi_output = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", 
                shell=True
            ).decode().strip()
            vram_mb = int(smi_output.split('\n')[0]) if smi_output else 0
            if vram_mb < 2048:
                print(f"Insufficient VRAM: {vram_mb}MB"); time.sleep(3)
                return False
        except Exception as e:
            print(f"VRAM check failed: {str(e)[:60]}"); time.sleep(1)
            return False

        # Check binary supports unified memory
        binary_path = Path("data/llama-cpp/main")
        if not binary_path.exists():
            print("Binary missing"); time.sleep(1)
            return False

        # Improved GPU support check using n-gpu-layers flag
        help_output = subprocess.check_output(f"{binary_path} --help", shell=True).decode()
        if "n-gpu-layers" not in help_output.lower():
            print("CUDA offload not enabled"); time.sleep(1)
            return False

        return True
    except Exception as e:
        print(f"CUDA check error: {str(e)[:60]}"); time.sleep(1)
        return False


def detect_hardware():
    """Detect system RAM and DDR level with better error handling"""
    try:
        # Get total system RAM from /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        temporary.SYSTEM_RAM_MB = int(parts[1]) // 1024
                        print(f"RAM: {temporary.SYSTEM_RAM_MB}MB"); time.sleep(1)
                    break
        
        # Get DDR level from dmidecode with permission handling
        try:
            output = subprocess.check_output(
                "sudo dmidecode --type memory 2>/dev/null | grep 'Type:'", 
                shell=True,
                stderr=subprocess.DEVNULL
            ).decode()
            if "DDR" in output:
                ddr_match = re.search(r'DDR(\d)', output)
                if ddr_match:
                    temporary.DDR_LEVEL = f"DDR{ddr_match.group(1)}"
                    print(f"DDR: {temporary.DDR_LEVEL}"); time.sleep(1)
            else:
                temporary.DDR_LEVEL = "Unknown"
        except:
            temporary.DDR_LEVEL = "Unknown"
            print("DDR: Unknown (run with sudo for details)"); time.sleep(1)
    except Exception as e:
        print(f"Hardware detect: {str(e)[:60]}"); time.sleep(1)
        temporary.SYSTEM_RAM_MB = 0
        temporary.DDR_LEVEL = "Unknown"

def main():
    try:
        print("Launcher initializing"); time.sleep(1)
        script_dir = Path(__file__).parent.resolve()
        os.chdir(script_dir)
        print("Directory set"); time.sleep(1)
        
        # Initialize data paths with absolute paths
        temporary.DATA_DIR = str(script_dir / "data")
        temporary.CONFIG_PATH = str(Path(temporary.DATA_DIR) / "persistent.json")
        temporary.HISTORY_DIR = str(Path(temporary.DATA_DIR) / "history")
        temporary.TEMP_DIR = str(Path(temporary.DATA_DIR) / "temp")
        temporary.LLAMA_CPP_BINARY = str(Path(temporary.DATA_DIR) / "llama-cpp" / "main")
        
        print(f"Data directory: {temporary.DATA_DIR}"); time.sleep(1)
        Path(temporary.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.LLAMA_CPP_BINARY).parent.mkdir(parents=True, exist_ok=True)
        
        # Hardware detection
        detect_hardware()
        
        # Pre-flight checks
        if not Path(temporary.LLAMA_CPP_BINARY).exists():
            print("Binary missing"); time.sleep(1)
            raise FileNotFoundError("llama.cpp binary not found")
        
        if not check_cuda_availability():
            print("CUDA not detected"); time.sleep(1)
            # Provide actionable advice instead of crashing
            print("Check installation with: nvcc --version && nvidia-smi")
            print("Verify binary supports CUDA: data/llama-cpp/main --help | grep 'n-gpu-layers'")
            raise RuntimeError("CUDA unavailable. See troubleshooting guide.")
        
        print("Loading config"); time.sleep(1)
        config_status = load_config()
        print("Config loaded"); time.sleep(1)
        
        print("Launching interface"); time.sleep(1)
        try:
            launch_interface()
        except Exception as e:
            print("Interface failed"); time.sleep(1)
            raise
    except Exception as e:
        print(f"Launcher error: {str(e)[:60]}"); time.sleep(3)
        # Don't crash - provide troubleshooting info
        print("\nTROUBLESHOOTING:")
        print("1. Verify CUDA installation: nvcc --version")
        print("2. Check GPU detection: nvidia-smi")
        print("3. Ensure installer ran successfully")
        print("4. Check binary exists: ls data/llama-cpp/main")
        print("5. Validate binary: data/llama-cpp/main --help")
        raise  # Re-raise after showing help

if __name__ == "__main__":
    main()