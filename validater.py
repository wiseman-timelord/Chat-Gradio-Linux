#!/usr/bin/env python3
# validater.py - Installation Validation Script
import os
import sys
import json
import re
import subprocess
import concurrent.futures
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Constants
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
BIN_DIR = BASE_DIR / "data" / "llama-bin"
CONFIG_PATH = BASE_DIR / "data" / "persistent.json"
MIN_CUDA_VERSION = 11.0
MIN_COMPUTE_CAPABILITY = 6.0

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass

def print_status(msg: str, success: Optional[bool] = None) -> None:
    """Improved status printer with colors"""
    if success is None:
        print(f"{YELLOW}•{RESET} {msg}")
    elif success:
        print(f"{GREEN}✓{RESET} {msg}")
    else:
        print(f"{RED}✗{RESET} {msg}")
    time.sleep(1 if success is None or success else 3)

def get_cuda_info(selected_gpu_index: int) -> Dict[str, any]:
    """Get detailed CUDA and GPU information for the selected GPU"""
    try:
        # Get CUDA version
        cuda_bin = "/usr/local/cuda/bin/nvcc"
        if not os.path.exists(cuda_bin):
            raise ValidationError("CUDA binary (nvcc) not found at /usr/local/cuda/bin/nvcc")
        nvcc = subprocess.check_output([cuda_bin, "--version"], text=True)
        cuda_match = re.search(r"release (\d+\.\d+)", nvcc)
        cuda_version = float(cuda_match.group(1)) if cuda_match else 0.0

        # Get GPU info
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        
        if not smi_output:
            raise ValidationError("No NVIDIA GPUs detected")

        # Find the selected GPU
        selected_gpu = None
        for line in smi_output.split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpu_index, gpu_name, driver_version, vram, compute_cap = parts
                if int(gpu_index) == selected_gpu_index:
                    selected_gpu = {
                        "cuda_version": cuda_version,
                        "driver_version": driver_version,
                        "gpu_name": gpu_name,
                        "vram_mb": int(vram),
                        "compute_capability": compute_cap,
                        "cuda_available": cuda_version >= MIN_CUDA_VERSION,
                        "unified_memory_capable": float(compute_cap) >= MIN_COMPUTE_CAPABILITY
                    }
                    break
        
        if selected_gpu is None:
            raise ValidationError(f"Selected GPU index {selected_gpu_index} not found")

        return selected_gpu
    except Exception as e:
        raise ValidationError(f"CUDA detection failed: {str(e)}")

def validate_environment() -> Tuple[bool, Dict[str, any]]:
    """Validate core environment setup"""
    print_status("Validating environment...", None)
    
    checks = {
        "venv_exists": VENV_DIR.exists(),
        "python_exists": (VENV_DIR / "bin" / "python").exists(),
        "llama_bin_exists": (BIN_DIR / "main").exists(),
        "config_exists": CONFIG_PATH.exists(),
        "data_temp_exists": (BASE_DIR / "data" / "temp").exists(),
        "data_history_exists": (BASE_DIR / "data" / "history").exists(),
        "data_vectors_exists": (BASE_DIR / "data" / "vectors").exists()
    }
    
    # Check Ubuntu version
    try:
        with open("/etc/os-release") as f:
            os_info = f.read()
        if "ubuntu" not in os_info.lower():
            print_status("  - Ubuntu OS", False)
            checks["ubuntu_os"] = False
        else:
            version_match = re.search(r'VERSION_ID="([\d.]+)"', os_info)
            ubuntu_version = float(version_match.group(1)) if version_match else 0.0
            checks["ubuntu_version"] = ubuntu_version >= 24.10
            print_status(f"  - Ubuntu version ({ubuntu_version})", checks["ubuntu_version"])
    except Exception:
        print_status("  - Ubuntu OS detection failed", False)
        checks["ubuntu_os"] = False
    
    # Get selected GPU index from config
    selected_gpu_index = 0
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
            selected_gpu_index = config["model_settings"].get("selected_gpu", 0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        print_status("  - Failed to read selected GPU from config", False)
        checks["config_read"] = False

    # Get CUDA info for selected GPU
    cuda_info = {}
    try:
        cuda_info = get_cuda_info(selected_gpu_index)
        checks["cuda_valid"] = cuda_info["cuda_available"]
        print_status(f"  - CUDA version ({cuda_info['cuda_version']})", checks["cuda_valid"])
        print_status(f"  - Unified memory support (GPU {selected_gpu_index})", cuda_info["unified_memory_capable"])
    except ValidationError as e:
        print_status(f"  - {str(e)}", False)
        checks["cuda_valid"] = False
    
    # Print remaining results
    for name, valid in checks.items():
        if name not in ["cuda_valid", "ubuntu_os", "ubuntu_version", "config_read"]:
            print_status(f"  - {name.replace('_', ' ')}", valid)
    
    return all(checks.values()), cuda_info

def validate_libraries() -> bool:
    """Validate all required Python libraries"""
    print_status("Validating libraries...", None)
    
    REQUIREMENTS = [
        ("gradio", "gradio", ">=4.25.0"),
        ("requests", "requests", "==2.31.0"),
        ("pyperclip", "pyperclip", ""),
        ("yake", "yake", ""),
        ("psutil", "psutil", ""),
        ("duckduckgo-search", "duckduckgo_search", ""),
        ("newspaper3k", "newspaper", ""),
        ("langchain-community", "langchain_community", "==0.3.18"),
        ("pygments", "pygments", "==2.17.2"),
        ("lxml_html_clean", "lxml_html_clean", ""),
        ("llama-cpp-python", "llama_cpp", "==0.2.23")
    ]
    
    venv_python = str(VENV_DIR / "bin" / "python")
    failed = []

    def test_library(pkg: Tuple[str, str, str]) -> Tuple[str, bool]:
        pkg_name, import_name, version = pkg
        try:
            cmd = [venv_python, "-c", f"import {import_name}; print('OK')"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return (pkg_name, result.returncode == 0 and "OK" in result.stdout)
        except Exception:
            return (pkg_name, False)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(test_library, REQUIREMENTS))
    
    for pkg_name, success in results:
        print_status(f"  - {pkg_name}", success)
        if not success:
            failed.append(pkg_name)
    
    if failed:
        print_status(f"  - {len(failed)} libraries failed", False)
        return False
    return True

def validate_config(cuda_info: Dict[str, any]) -> bool:
    """Validate configuration file"""
    print_status("Validating configuration...", None)
    
    if not CONFIG_PATH.exists():
        print_status("  - Config file missing", False)
        return False
    
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        
        checks = [
            ("llama binary path", (BASE_DIR / config["model_settings"]["llama_cli_path"]).exists()),
            ("use python bindings", config["model_settings"].get("use_python_bindings", False) == True),
            ("unified memory enabled", config["model_settings"].get("unified_memory", False) == True),
            ("backend type", config["backend_config"].get("backend_type", "") == "CUDA"),
            ("VRAM allocation", 
             config["model_settings"]["vram_size"] <= cuda_info.get("vram_mb", float('inf'))),
            ("config field model_dir", "model_dir" in config["model_settings"]),
            ("config field llama_cli_path", "llama_cli_path" in config["model_settings"]),
            ("config field vram_size", "vram_size" in config["model_settings"]),
            ("config field n_batch", "n_batch" in config["model_settings"]),
            ("config field dynamic_gpu_layers", "dynamic_gpu_layers" in config["model_settings"])
        ]
        
        all_valid = True
        for name, valid in checks:
            print_status(f"  - {name}", valid)
            if not valid:
                all_valid = False
        
        return all_valid
        
    except json.JSONDecodeError:
        print_status("  - Invalid JSON format", False)
        return False
    except Exception as e:
        print_status(f"  - Config error: {str(e)}", False)
        return False

def validate_cuda_integration() -> bool:
    """Test actual CUDA functionality with llama-cpp-python"""
    print_status("Testing CUDA integration...", None)
    
    test_script = """
import llama_cpp
import sys

try:
    llama_cpp.llama_backend_init()
    
    n_gpu_layers = 1
    model_params = llama_cpp.llama_model_default_params()
    model_params.n_gpu_layers = n_gpu_layers
    model_params.use_mmap = False
    
    ctx = llama_cpp.llama_new_context_with_model(None, model_params)
    if ctx is not None:
        print("CUDA_TEST:Success")
    else:
        print("CUDA_TEST:Failed")
except Exception as e:
    print(f"ERROR:{str(e)}")
finally:
    llama_cpp.llama_backend_free()
"""
    
    venv_python = str(VENV_DIR / "bin" / "python")
    try:
        result = subprocess.run(
            [venv_python, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout.strip()
        if "ERROR:" in output:
            print_status(f"  - Error: {output.split('ERROR:')[1]}", False)
            return False
        
        checks = [
            ("CUDA initialization", "CUDA_TEST:Success" in output)
        ]
        
        all_valid = True
        for name, valid in checks:
            print_status(f"  - {name}", valid)
            if not valid:
                all_valid = False
        
        return all_valid
        
    except subprocess.TimeoutExpired:
        print_status("  - Test timed out", False)
        return False
    except Exception as e:
        print_status(f"  - Test failed: {str(e)}", False)
        return False

def main():
    try:
        env_ok, cuda_info = validate_environment()
        if not env_ok:
            raise ValidationError("Environment validation failed")
        
        if not validate_libraries():
            raise ValidationError("Library validation failed")
        
        if not validate_config(cuda_info):
            raise ValidationError("Configuration validation failed")
        
        if cuda_info.get("cuda_available", False):
            if not validate_cuda_integration():
                raise ValidationError("CUDA integration test failed")
        
        print_status("\nValidation successful!", True)
        if sys.stdin.isatty():
            input("Press Enter to continue...")
        return 0
        
    except ValidationError as e:
        print_status(f"\nValidation failed: {str(e)}", False)
        if sys.stdin.isatty():
            input("Press Enter to continue...")
        return 1
    except Exception as e:
        print_status(f"\nUnexpected error: {str(e)}", False)
        if sys.stdin.isatty():
            input("Press Enter to continue...")
        return 1

if __name__ == "__main__":
    sys.exit(main())