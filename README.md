# Chat-Linux-Gguf
Status: Alpha - Issues with installer (the reason I use pre-built wheels normally, see `Installer Issues` below).

### Description
Chat-Linux-Gguf is the Ubuntu version of [Chat-Gradio-Gguf](https://github.com/wiseman-timelord/Chat-Gradio-Gguf) (which runs on Windows). So, the plan for the differences are...
- .sh instead of .bat, as well as program the scripts for specifically linux/ubuntu.
- Cuda Only, gpu must still be selectable. Installer will only have options for Cuda 11 and Cuda 12. There will also be some optimization/cleanup after
- The user must install cuda toolkit themselves.
- The scripts MUST use unified memory, so as to do the processing on the selected card, while loading models to system+gpu memory optimally. There will be no requirement for calculating how many layers to load to the GPU, we will assume that the models will load on the system memory available. optimize/cleanup functions/processes for removal of the calculations.

### Demonstration
- The `Bash Menu` is looking good...
```
===============================================================================
    Chat-Linux-Gguf: Bash Menu
===============================================================================






    1. Run Main Program

    2. Run Installation

    3. Run Validation






===============================================================================
Selection; Menu Options = 1-3, Exit Bash = X: 

```
- Installer currently (this is why the program was not done in time for solstice)...
```
===============================================================================
    Chat-Linux-Gguf: Installer
===============================================================================

Note: Installation may require sudo for system dependencies.
WARNING: This will delete existing ./data and ./.venv directories.
Continue? (y/N): y
Preparing installation...
[2025-06-21 12:48:12] INFO: Deleted: data
[2025-06-21 12:48:14] INFO: Deleted: .venv
Starting installer...
[2025-06-21 12:48:16] INFO: Installing Chat-Linux-Gguf
[2025-06-21 12:48:17] INFO: Verifying system...
[2025-06-21 12:48:18] INFO: Detected architectures: 61
[2025-06-21 12:48:19] INFO: Detected GPUs:
[2025-06-21 12:48:19] INFO: [0] NVIDIA GeForce GTX 1060 3GB ✓UM VRAM: 3072MB
[2025-06-21 12:48:20] INFO: Auto-selected GPU: 0
[2025-06-21 12:48:21] INFO: System Summary:
[2025-06-21 12:48:21] INFO: CUDA: 11.0
[2025-06-21 12:48:21] INFO: Driver: 570.133.07
[2025-06-21 12:48:21] INFO: GPU: 0
[2025-06-21 12:48:22] INFO: Installing system dependencies
[2025-06-21 12:48:23] INFO: Cleaning problematic repositories
[2025-06-21 12:48:23] WARNING: CUDA repo version mismatch detected
[2025-06-21 12:48:23] WARNING: Consider updating CUDA keyring
[2025-06-21 12:48:24] INFO: Installing software-properties-common
[2025-06-21 12:48:24] INFO: software-properties-common installed
[2025-06-21 12:48:25] INFO: Adding GCC Toolchain PPA
[2025-06-21 12:48:31] INFO: PPA added
[2025-06-21 12:48:32] INFO: Enabling universe repository
[2025-06-21 12:48:37] INFO: Universe enabled
[2025-06-21 12:48:38] INFO: Updating package lists
[2025-06-21 12:48:40] WARNING: Toolchain PPA unavailable, continuing
[2025-06-21 12:48:41] INFO: Package lists updated
[2025-06-21 12:48:42] INFO: Installing build tools
[2025-06-21 12:48:43] INFO: Build tools installed
[2025-06-21 12:48:44] INFO: Installing GCC 9
[2025-06-21 12:48:44] INFO: Configuring GCC alternatives
[2025-06-21 12:48:44] INFO: Active GCC: gcc (Ubuntu 9.5.0-6ubuntu2) 9.5.0
[2025-06-21 12:48:45] INFO: Verifying cmake installation
[2025-06-21 12:48:45] INFO: CMake verified
[2025-06-21 12:48:46] INFO: Installing dev libraries
[2025-06-21 12:48:47] INFO: Dev libraries installed
[2025-06-21 12:48:48] INFO: Installing Python dev
[2025-06-21 12:48:49] INFO: Python dev installed
[2025-06-21 12:48:50] INFO: Installing utilities
[2025-06-21 12:48:50] INFO: Utilities installed
[2025-06-21 12:48:51] INFO: System dependencies completed
[2025-06-21 12:48:52] INFO: Setting up llama.cpp
[2025-06-21 12:48:53] INFO: Cloning attempt 1/3
[==========] 100% 48.8MB/48.8MB 57s/57s
[2025-06-21 12:49:50] INFO: Clone complete
[2025-06-21 12:49:51] INFO: Compiling llama.cpp...
[2025-06-21 12:49:52] INFO: Checking compiler compatibility
[2025-06-21 12:49:53] INFO: CUDA 11.0 max gcc: 9
[2025-06-21 12:49:54] INFO: Using gcc-9
[2025-06-21 12:49:55] INFO: Compiler: gcc-9 (Ubuntu 9.5.0-6ubuntu2) 9.5.0
[2025-06-21 12:49:56] INFO: Applied patch for __builtin_assume
[2025-06-21 12:49:57] INFO: Configuring CMake build
[2025-06-21 12:50:00] INFO: CMake configured successfully
[2025-06-21 12:50:01] INFO: Compiling binary...
[2025-06-21 12:51:55] INFO: Make output: [  0%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o
...

...
[2025-06-21 12:51:55] INFO: Binary compiled successfully
[2025-06-21 12:51:56] INFO: Installing binary...
[2025-06-21 12:51:56] INFO: Binary installed from /media/mastar/Progs_250/Programs/Chat-Linux-Gguf/Chat-Linux-Gguf-A069/data/temp/llama.cpp/build/bin/llama-cli to /media/mastar/Progs_250/Programs/Chat-Linux-Gguf/Chat-Linux-Gguf-A069/data/llama-cpp/llama
[2025-06-21 12:51:57] INFO: Testing binary...
[2025-06-21 12:51:57] INFO: Binary test passed
[2025-06-21 12:51:58] INFO: Creating Python env...
[2025-06-21 12:51:59] INFO: Creating virtual env
[2025-06-21 12:52:01] INFO: Virtual env created
[2025-06-21 12:52:02] INFO: Upgrading pip...
[2025-06-21 12:54:02] ERROR: Pip upgrade failed: Timeout after 120s
[2025-06-21 12:54:05] ERROR: Failed: Pip upgrade failed
[2025-06-21 12:54:05] INFO: Cleaning up...
[2025-06-21 12:54:06] ERROR: Installation failed (code: 1)
[2025-06-21 12:54:09] INFO: Virtual environment status reset

```

### Requirements
- O.S. - Linux Only (Ubuntu ~24.10 recommended).
- Python - Unknown min version (will be assessed after working version, but presumed 3.8+).
- G.P.U. - nVidia with nVidia driver 450.80+, may be used/compute GPU. 
- Cuda Toolkit - You must install, Version [11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive), depending upon Cuda level of card for processing.
- R.A.M. - Models are stored in Unified Memory, and processes on the GPU. This method should cover most model cases.

### Development
- Need to complete "./installer.py" script.
- we will want an additional bar showing `GPU Ram Usage / System Ram Usage`. Somehow travelling bars on top and bottom, for memory/but this would be using frames, is this possible?

### File Structure
Details of file structure...
- Core Project files...
```
project_root/
│ Chat-Linux-Gguf.sh
│ installer.py
│ validater.py
│ launcher.py
├── media/
│ └── project_banner.jpg
├── scripts/
│ └── interface.py
│ └── models.py
│ └── prompts.py
│ └── settings.py
│ └── temporary.py
│ └── utlity.py
```
- Installed/Temporary files...
```
project_root/
├── data/
│ └── persistent.json
├── data/vectors/
└─────── *
├── data/temp/
└────── *
├── data/history
└────── *
├── .venv/ (if we are still using this)
└────── *
```

### Notation
- Ubuntu ChatBots now? In-short Ubuntu 24.10 has allows to install, "RX 470" as main with "1060 3GB" as compute for cuda, with no issues, thanks to help from [X-Grok](www.x.com).
- Produced via manually prompting AI with NotePadNext v0.12, making use of AI Systems, Grok, X-Grok, DeepSeek, Claude.
