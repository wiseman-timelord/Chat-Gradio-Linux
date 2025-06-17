#!/usr/bin/env python3
# Enhanced Installer for Chat-Linux-Gguf

import os
import json
import platform
import subprocess
import sys
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Constants
APP_NAME = "Chat-Linux-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
LLAMA_DIR = BASE_DIR / "llama.cpp"
BUILD_DIR = LLAMA_DIR / "build"
BIN_DIR = BASE_DIR / "data" / "llama-bin"
CONFIG_PATH = BASE_DIR / "data" / "persistent.json"

# Minimum requirements
MIN_CUDA_VERSION = 11.0
MIN_DRIVER_VERSION = 450.80
MIN_VRAM = 2048  # 2GB minimum VRAM
MIN_COMPUTE_CAPABILITY = 6.0  # Minimum compute capability for unified memory

REQUIREMENTS = [
    "gradio>=4.25.0",
    "requests==2.31.0",
    "pyperclip",
    "yake",
    "psutil",
    "duckduckgo-search",
    "newspaper3k",
    "langchain-community==0.3.18",
    "pygments==2.17.2",
    "lxml_html_clean",
    "llama-cpp-python==0.2.23",
]

CONFIG_TEMPLATE = """{
    "model_settings": {
        "model_dir": "models",
        "model_name": "",
        "context_size": 8192,
        "temperature": 0.66,
        "repeat_penalty": 1.1,
        "use_python_bindings": true,
        "llama_cli_path": "data/llama-bin/main",
        "vram_size": %d,
        "selected_gpu": %d,
        "mmap": false,
        "mlock": false,
        "n_batch": %d,
        "dynamic_gpu_layers": false,
        "max_history_slots": 12,
        "max_attach_slots": 6,
        "session_log_height": 500,
        "unified_memory": true,
        "gpu_layers": -1
    },
    "backend_config": {
        "backend_type": "CUDA",
        "llama_bin_path": "data/llama-bin",
        "cuda_version": "%s",
        "compute_capability": "%s",
        "available_gpus": %s
    }
}"""

class InstallerError(Exception):
    """Custom exception for installer failures"""
    pass

def log_message(message: str, level: str = "INFO") -> None:
    """Log messages to console only"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def run_command(cmd: list, cwd: Optional[Path] = None, timeout: int = 300, 
                env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """Execute a command with error handling and timeout"""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"

def get_cuda_compute_capabilities() -> List[str]:
    """Get compute capabilities of all GPUs"""
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        
        capabilities = []
        for line in smi_output.split('\n'):
            if line.strip():
                cap = line.strip()
                if cap not in capabilities:
                    capabilities.append(cap)
        
        return capabilities
        
    except Exception as e:
        log_message(f"Could not detect compute capabilities: {e}", "WARNING")
        time.sleep(1)
        return []

def get_cuda_architecture_flags() -> Tuple[List[str], str]:
    """Detect CUDA architectures for compilation and return flags + arch string"""
    capabilities = get_cuda_compute_capabilities()
    
    if not capabilities:
        log_message("Using fallback architectures", "WARNING")
        time.sleep(1)
        arch_string = "60;61;70;75;80;86;89;90"
        return [f"-DCMAKE_CUDA_ARCHITECTURES={arch_string}"], arch_string
    
    architectures = []
    for cap in capabilities:
        try:
            arch = cap.replace('.', '')
            if len(arch) == 2:  # Ensure major.minor format
                architectures.append(arch)
        except ValueError:
            continue
    
    if not architectures:
        arch_string = "60;61;70;75;80;86;89;90"
    else:
        architectures = sorted(list(set(architectures)))
        arch_string = ";".join(architectures)
    
    log_message(f"Detected CUDA architectures: {arch_string}")
    time.sleep(1)
    return [f"-DCMAKE_CUDA_ARCHITECTURES={arch_string}"], arch_string

def check_unified_memory_support(compute_cap: str) -> bool:
    """Check if GPU supports unified memory effectively"""
    try:
        cap_float = float(compute_cap)
        return cap_float >= MIN_COMPUTE_CAPABILITY
    except (ValueError, TypeError):
        return False

def check_cuda_compatibility(cuda_version: str, gpus: List[Dict[str, any]]) -> bool:
    """Check if CUDA version is compatible with GPUs"""
    cuda_major = float(cuda_version.split('.')[0])
    for gpu in gpus:
        compute_cap = float(gpu["compute_capability"])
        # Simplified compatibility: CUDA 11 supports compute 5.0+, CUDA 12 supports 6.0+
        if cuda_major >= 12 and compute_cap < 6.0:
            log_message(f"CUDA {cuda_version} may not be compatible with GPU {gpu['name']} (compute {compute_cap})", "WARNING")
            time.sleep(1)
            return False
        if cuda_major < 12 and compute_cap >= 8.0:
            log_message(f"CUDA {cuda_version} may not be optimal for GPU {gpu['name']} (compute {compute_cap})", "WARNING")
            time.sleep(1)
    return True

def select_gpu(gpus: List[Dict[str, any]]) -> int:
    """Prompt user to select a GPU"""
    log_message("Detected GPUs:")
    for gpu in gpus:
        unified = "✓UM" if gpu["unified_memory_capable"] else "✗UM"
        log_message(f"[{gpu['index']}] {gpu['name']} - {gpu['vram']}MB VRAM - Compute {gpu['compute_capability']} {unified}")
    time.sleep(1)
    
    while True:
        try:
            if sys.stdin.isatty():
                choice = input("Select GPU by index (default 0): ").strip()
                choice = int(choice) if choice else 0
                if any(gpu['index'] == choice for gpu in gpus):
                    log_message(f"Selected GPU: {choice}")
                    time.sleep(1)
                    return choice
                log_message(f"Invalid GPU index: {choice}", "ERROR")
                time.sleep(3)
            else:
                # Non-interactive: select GPU with most VRAM
                best_gpu = max(gpus, key=lambda x: x["vram"])
                log_message(f"Non-interactive mode: Selected GPU {best_gpu['index']}")
                time.sleep(1)
                return best_gpu['index']
        except ValueError:
            log_message("Invalid input. Please enter a valid GPU index.", "ERROR")
            time.sleep(3)

def check_system() -> Tuple[bool, Dict[str, any]]:
    """Verify system meets all requirements"""
    system_info = {
        "cuda_version": None,
        "driver_version": None,
        "gpus": [],
        "selected_gpu": 0,
        "unified_memory_support": False,
        "cuda_architectures": []
    }

    log_message("Verifying system requirements...")
    time.sleep(1)

    # Check Linux distribution
    if platform.system() != "Linux":
        log_message("This installer is for Linux only", "ERROR")
        time.sleep(3)
        raise InstallerError("This installer is for Linux only")

    # Check for Ubuntu
    try:
        with open("/etc/os-release") as f:
            os_info = f.read()
        if "ubuntu" not in os_info.lower():
            log_message("Warning: This installer is optimized for Ubuntu 24.10", "WARNING")
            time.sleep(1)
        else:
            version_match = re.search(r'VERSION_ID="([\d.]+)"', os_info)
            if version_match and float(version_match.group(1)) < 24.10:
                log_message("Warning: Ubuntu version below 24.10 detected", "WARNING")
                time.sleep(1)
    except Exception:
        log_message("Could not verify Ubuntu version", "WARNING")
        time.sleep(1)

    # Check NVIDIA drivers and GPUs
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,driver_version,memory.total,name,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        
        if not smi_output:
            log_message("No NVIDIA GPUs detected", "ERROR")
            time.sleep(3)
            raise InstallerError("No NVIDIA GPUs detected")

        gpus = []
        driver_version = None
        
        for line in smi_output.split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    try:
                        gpu_index, driver_ver, vram, gpu_name, compute_cap = parts
                        if driver_version is None:
                            driver_version = driver_ver
                        
                        gpu_info = {
                            "index": int(gpu_index),
                            "vram": int(vram),
                            "name": gpu_name,
                            "compute_capability": compute_cap,
                            "unified_memory_capable": check_unified_memory_support(compute_cap)
                        }
                        gpus.append(gpu_info)
                    except (ValueError, IndexError) as e:
                        log_message(f"Failed to parse GPU info: {line} - {e}", "WARNING")
                        time.sleep(1)
                        continue

        if not gpus:
            log_message("No valid NVIDIA GPUs found", "ERROR")
            time.sleep(3)
            raise InstallerError("No valid NVIDIA GPUs found")

        # Check driver version
        try:
            driver_parts = driver_version.split('.')
            driver_num = float(f"{driver_parts[0]}.{driver_parts[1]}")
            if driver_num < MIN_DRIVER_VERSION:
                log_message(f"Driver version {MIN_DRIVER_VERSION}+ required (found {driver_version})", "ERROR")
                time.sleep(3)
                raise InstallerError(f"Driver version {MIN_DRIVER_VERSION}+ required (found {driver_version})")
        except (ValueError, IndexError):
            log_message(f"Could not parse driver version: {driver_version}", "WARNING")
            time.sleep(1)

        system_info.update({
            "driver_version": driver_version,
            "gpus": gpus,
            "unified_memory_support": any(gpu["unified_memory_capable"] for gpu in gpus)
        })

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log_message("NVIDIA drivers not properly installed or nvidia-smi not found", "ERROR")
        time.sleep(3)
        raise InstallerError("NVIDIA drivers not properly installed or nvidia-smi not found")

    # Check CUDA version and binary
    try:
        cuda_bin = "/usr/local/cuda/bin/nvcc"
        if not os.path.exists(cuda_bin):
            log_message("CUDA binary (nvcc) not found at /usr/local/cuda/bin/nvcc", "ERROR")
            time.sleep(3)
            raise InstallerError("CUDA binary (nvcc) not found at /usr/local/cuda/bin/nvcc")
        nvcc = subprocess.check_output([cuda_bin, "--version"], text=True)
        match = re.search(r"release (\d+\.\d+)", nvcc)
        if not match:
            log_message("CUDA version detection failed", "ERROR")
            time.sleep(3)
            raise InstallerError("CUDA version detection failed")
        
        cuda_version = float(match.group(1))
        if cuda_version < MIN_CUDA_VERSION:
            log_message(f"CUDA {MIN_CUDA_VERSION}+ required (found {cuda_version})", "ERROR")
            time.sleep(3)
            raise InstallerError(f"CUDA {MIN_CUDA_VERSION}+ required (found {cuda_version})")
        
        system_info["cuda_version"] = match.group(1)

        # Check CUDA compatibility with GPUs
        if not check_cuda_compatibility(system_info["cuda_version"], gpus):
            log_message("CUDA version may not be compatible with detected GPUs", "ERROR")
            time.sleep(3)
            raise InstallerError("CUDA version may not be compatible with detected GPUs")

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log_message("CUDA Toolkit not found - please install CUDA 11.0+", "ERROR")
        time.sleep(3)
        raise InstallerError("CUDA Toolkit not found - please install CUDA 11.0+")

    # Get CUDA architectures
    _, arch_string = get_cuda_architecture_flags()
    system_info["cuda_architectures"] = arch_string

    # Select GPU
    suitable_gpus = [gpu for gpu in gpus if gpu["vram"] >= MIN_VRAM and gpu["unified_memory_capable"]]
    if not suitable_gpus:
        suitable_gpus = [gpu for gpu in gpus if gpu["vram"] >= MIN_VRAM]
        if not suitable_gpus:
            log_message(f"No GPUs with sufficient VRAM ({MIN_VRAM}MB)", "ERROR")
            time.sleep(3)
            raise InstallerError(f"No GPUs with sufficient VRAM ({MIN_VRAM}MB)")
        log_message("Warning: Selected GPU may not have optimal unified memory support", "WARNING")
        time.sleep(1)
    
    system_info["selected_gpu"] = select_gpu(suitable_gpus)

    return True, system_info

def install_system_deps() -> bool:
    """Install required system dependencies"""
    log_message("Installing system dependencies...")
    time.sleep(1)
    
    deps = [
        "git", "cmake", "build-essential", "pkg-config",
        "libssl-dev", "zlib1g-dev", "libbz2-dev",
        "libreadline-dev", "libsqlite3-dev", "wget",
        "curl", "llvm", "libncurses5-dev", "xz-utils",
        "tk-dev", "libxml2-dev", "libxmlsec1-dev",
        "libffi-dev", "liblzma-dev", "python3-dev"
    ]
    
    success, output = run_command(["sudo", "apt", "update"], timeout=120)
    if not success:
        log_message(f"Failed to update packages:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"Failed to update packages:\n{output}")
    
    success, output = run_command(["sudo", "apt", "install", "-y"] + deps, timeout=300)
    if not success:
        log_message(f"Failed to install dependencies:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"Failed to install dependencies:\n{output}")
    
    time.sleep(1)
    return True

def setup_llama_cpp() -> bool:
    """Clone and prepare llama.cpp supper"""
    log_message("Setting up llama.cpp...")
    time.sleep(1)
    
    if LLAMA_DIR.exists():
        log_message("Removing existing llama.cpp directory...")
        time.sleep(1)
        shutil.rmtree(LLAMA_DIR)
    
    success, output = run_command([
        "git", "clone", "--depth", "1",
        "https://github.com/ggerganov/llama.cpp.git",
        str(LLAMA_DIR)
    ], timeout=180)
    
    if not success:
        log_message(f"Failed to clone llama.cpp:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"Failed to clone llama.cpp:\n{output}")
    
    time.sleep(1)
    return True

def compile_llama_cpp(cuda_version: str, arch_string: str) -> bool:
    """Compile llama.cpp with CUDA and unified memory support"""
    log_message("Compiling llama.cpp with CUDA and unified memory support...")
    time.sleep(1)
    
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        time.sleep(1)
    BUILD_DIR.mkdir(parents=True)
    
    arch_flags, _ = get_cuda_architecture_flags()
    
    cmake_flags = [
        "-DLLAMA_CUDA=ON",
        "-DLLAMA_CUDA_UNIFIED_MEMORY=ON",
        "-DLLAMA_CUDA_F16=ON",
        "-DLLAMA_CUDA_MMV_Y=8",
        "-DLLAMA_CUDA_PEER_MAX_BATCH_SIZE=128",
        "-DLLAMA_CUDA_DMMV_X=32",
        "-DLLAMA_CUDA_DMMV_Y=1",
        "-DLLAMA_CUDA_KQUANTS_ITER=2",
        "-DLLAMA_CUDA_FORCE_DMMV=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_NATIVE=OFF",
        "-DLLAMA_STATIC=OFF"
    ] + arch_flags
    
    cuda_env = os.environ.copy()
    cuda_env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
    })
    
    success, output = run_command(
        ["cmake", ".."] + cmake_flags,
        cwd=BUILD_DIR,
        timeout=180,
        env=cuda_env
    )
    if not success:
        log_message(f"CMake configuration failed:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"CMake configuration failed:\n{output}")
    
    cpu_count = min(os.cpu_count() or 4, 8)
    success, output = run_command(
        ["make", "-j", str(cpu_count), "main"],
        cwd=BUILD_DIR,
        timeout=900,
        env=cuda_env
    )
    if not success:
        log_message(f"Compilation failed:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"Compilation failed:\n{output}")
    
    possible_locations = [
        BUILD_DIR / "bin" / "main",
        BUILD_DIR / "main"
    ]
    
    main_binary = None
    for location in possible_locations:
        if location.exists():
            main_binary = location
            break
    
    if not main_binary:
        log_message("Compilation failed - binary not found in expected locations", "ERROR")
        time.sleep(3)
        raise InstallerError("Compilation failed - binary not found in expected locations")
    
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(main_binary, BIN_DIR / "main")
    
    os.chmod(BIN_DIR / "main", 0o755)
    
    success, output = run_command([str(BIN_DIR / "main"), "--help"], timeout=10)
    if not success:
        log_message(f"Binary test failed: {output}", "WARNING")
        time.sleep(1)
    
    time.sleep(1)
    return True

def setup_python_env(cuda_version: str, arch_string: str) -> bool:
    """Set up Python virtual environment and dependencies"""
    log_message("Creating Python virtual environment...")
    time.sleep(1)
    
    if VENV_DIR.exists():
        log_message("Removing existing virtual environment...")
        time.sleep(1)
        shutil.rmtree(VENV_DIR)
    
    success, output = run_command([sys.executable, "-m", "venv", str(VENV_DIR)])
    if not success:
        log_message(f"Virtual environment creation failed:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"Virtual environment creation failed:\n{output}")
    
    pip = str(VENV_DIR / "bin" / "pip")
    success, output = run_command([pip, "install", "--upgrade", "pip", "wheel", "setuptools"], timeout=120)
    if not success:
        log_message(f"Pip upgrade failed:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"Pip upgrade failed:\n{output}")
    
    log_message("Installing Python packages...")
    time.sleep(1)
    success, output = run_command([pip, "install"] + REQUIREMENTS, timeout=600)
    if not success:
        log_message(f"Package installation failed:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"Package installation failed:\n{output}")
    
    log_message("Installing llama-cpp-python with CUDA and unified memory support...")
    time.sleep(1)
    llama_cpp_env = os.environ.copy()
    llama_cpp_env.update({
        "LLAMA_CUDA": "1",
        "CMAKE_ARGS": f"-DLLAMA_CUDA=ON -DLLAMA_CUDA_UNIFIED_MEMORY=ON -DCMAKE_CUDA_ARCHITECTURES={arch_string}",
        "FORCE_CMAKE": "1",
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
    })
    
    success, output = run_command([
        pip, "install",
        "--force-reinstall",
        "--no-cache-dir",
        "llama-cpp-python==0.2.23"
    ], timeout=1200, env=llama_cpp_env)
    
    if not success:
        log_message(f"llama-cpp-python installation failed:\n{output}", "ERROR")
        time.sleep(3)
        raise InstallerError(f"llama-cpp-python installation failed:\n{output}")
    
    time.sleep(1)
    return True

def create_config(system_info: Dict[str, any]) -> None:
    """Create configuration file with detected settings"""
    log_message("Creating configuration...")
    time.sleep(1)
    
    selected_gpu = system_info["selected_gpu"]
    gpu_info = next((gpu for gpu in system_info["gpus"] if gpu["index"] == selected_gpu), system_info["gpus"][0])
    
    vram_mb = gpu_info["vram"]
    if vram_mb >= 12288:
        n_batch = 8192
    elif vram_mb >= 8192:
        n_batch = 4096
    elif vram_mb >= 4096:
        n_batch = 2048
    else:
        n_batch = 1024
    
    for dir_path in [BASE_DIR / "data", BASE_DIR / "data" / "temp", 
                     BASE_DIR / "data" / "history", BASE_DIR / "data" / "vectors"]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        f.write(CONFIG_TEMPLATE % (
            vram_mb,
            selected_gpu,
            n_batch,
            system_info["cuda_version"],
            gpu_info["compute_capability"],
            json.dumps(system_info["gpus"])
        ))
    
    if not CONFIG_PATH.exists():
        log_message("Failed to create configuration file", "ERROR")
        time.sleep(3)
        raise InstallerError("Failed to create configuration file")
    
    try:
        with open(CONFIG_PATH, "r") as f:
            json.load(f)
    except json.JSONDecodeError:
        log_message("Created configuration file is invalid JSON", "ERROR")
        time.sleep(3)
        raise InstallerError("Created configuration file is invalid JSON")
    
    time.sleep(1)

def print_system_summary(system_info: Dict[str, any]) -> None:
    """Print installation summary"""
    log_message("\n" + "="*60)
    log_message("INSTALLATION SUMMARY")
    log_message("="*60)
    log_message(f"CUDA Version: {system_info['cuda_version']}")
    log_message(f"Driver Version: {system_info['driver_version']}")
    log_message(f"CUDA Architectures: {system_info['cuda_architectures']}")
    log_message(f"Selected GPU: {system_info['selected_gpu']}")
    
    for gpu in system_info["gpus"]:
        marker = " (SELECTED)" if gpu["index"] == system_info["selected_gpu"] else ""
        unified = " ✓UM" if gpu["unified_memory_capable"] else " ✗UM"
        log_message(f"GPU {gpu['index']}: {gpu['name']} - {gpu['vram']}MB VRAM - Compute {gpu['compute_capability']}{unified}{marker}")
    
    log_message(f"Unified Memory Support: {'Yes' if system_info['unified_memory_support'] else 'No'}")
    log_message("="*60)
    time.sleep(1)

def cleanup_on_failure():
    """Clean up partial installation on failure"""
    log_message("Cleaning up partial installation...", "WARNING")
    time.sleep(1)
    paths_to_clean = [VENV_DIR, LLAMA_DIR, BUILD_DIR, BIN_DIR]
    for path in paths_to_clean:
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception as e:
                log_message(f"Failed to clean {path}: {e}", "WARNING")
                time.sleep(1)

def main():
    try:
        log_message(f"Starting {APP_NAME} installation")
        log_message(f"System: {platform.platform()}")
        log_message(f"Python: {sys.version}")
        time.sleep(1)
        
        if sys.stdin.isatty():
            log_message("WARNING: This will delete ./data and ./.venv directories.")
            response = input("Continue? (y/N): ").strip().lower()
            if response != 'y':
                log_message("Installation cancelled by user", "INFO")
                time.sleep(1)
                sys.exit(0)
            time.sleep(1)
        
        try:
            sys_ok, system_info = check_system()
        except InstallerError as e:
            log_message(f"System check failed: {str(e)}", "ERROR")
            time.sleep(3)
            sys.exit(1)
        
        print_system_summary(system_info)
        
        if not system_info["unified_memory_support"]:
            log_message("WARNING: No GPUs with optimal unified memory support detected", "WARNING")
            log_message("Performance may be suboptimal", "WARNING")
            time.sleep(1)
            if sys.stdin.isatty():
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    log_message("Installation cancelled by user", "INFO")
                    time.sleep(1)
                    sys.exit(0)
                time.sleep(1)
        
        if not install_system_deps():
            log_message("System dependency installation failed", "ERROR")
            time.sleep(3)
            raise InstallerError("System dependency installation failed")
        
        if not setup_llama_cpp():
            log_message("llama.cpp setup failed", "ERROR")
            time.sleep(3)
            raise InstallerError("llama.cpp setup failed")
        
        if not compile_llama_cpp(system_info['cuda_version'], system_info['cuda_architectures']):
            log_message("Compilation failed", "ERROR")
            time.sleep(3)
            raise InstallerError("Compilation failed")
        
        if not setup_python_env(system_info['cuda_version'], system_info['cuda_architectures']):
            log_message("Python environment setup failed", "ERROR")
            time.sleep(3)
            raise InstallerError("Python environment setup failed")
        
        create_config(system_info)
        
        log_message("\n" + "="*60)
        log_message("INSTALLATION COMPLETED SUCCESSFULLY!")
        log_message("="*60)
        log_message(f"To run: {VENV_DIR}/bin/python launcher.py")
        log_message("Run 'python validater.py' to verify installation")
        log_message("="*60)
        time.sleep(1)
        
        if sys.stdin.isatty():
            input("Press Enter to continue...")
        
    except InstallerError as e:
        log_message(f"\nINSTALLATION FAILED: {str(e)}", "ERROR")
        time.sleep(3)
        cleanup_on_failure()
        sys.exit(1)
    except KeyboardInterrupt:
        log_message("\nInstallation cancelled by user", "WARNING")
        time.sleep(1)
        cleanup_on_failure()
        sys.exit(1)
    except Exception as e:
        log_message(f"\nUNEXPECTED ERROR: {str(e)}", "ERROR")
        import traceback
        log_message(traceback.format_exc(), "ERROR")
        time.sleep(3)
        cleanup_on_failure()
        sys.exit(1)

if __name__ == "__main__":
    main()