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
- Installer issues currently...
```
===============================================================================
    Chat-Linux-Gguf: Installer
===============================================================================

Note: Installation may require sudo for system dependencies.
WARNING: This will delete existing ./data and ./.venv directories.
Continue? (y/N): y
Preparing installation...
Starting installer...
[2025-06-20 23:44:17] INFO: Installing Chat-Linux-Gguf
[2025-06-20 23:44:18] INFO: Verifying system...
[2025-06-20 23:44:19] INFO: Detected architectures: 61
[2025-06-20 23:44:20] INFO: Detected GPUs:
[2025-06-20 23:44:20] INFO: [0] NVIDIA GeForce GTX 1060 3GB ✓UM VRAM: 3072MB
[2025-06-20 23:44:21] INFO: Auto-selected GPU: 0
[2025-06-20 23:44:22] INFO: System Summary:
[2025-06-20 23:44:22] INFO: CUDA: 11.0
[2025-06-20 23:44:22] INFO: Driver: 570.133.07
[2025-06-20 23:44:22] INFO: GPU: 0
[2025-06-20 23:44:23] INFO: Installing system dependencies
[2025-06-20 23:44:24] INFO: Cleaning problematic repositories
[2025-06-20 23:44:24] WARNING: CUDA repo version mismatch detected
[2025-06-20 23:44:24] WARNING: Consider updating CUDA keyring
[2025-06-20 23:44:25] INFO: Installing software-properties-common
[2025-06-20 23:44:26] INFO: software-properties-common installed
[2025-06-20 23:44:27] INFO: Adding GCC Toolchain PPA
[2025-06-20 23:44:39] INFO: PPA added
[2025-06-20 23:44:40] INFO: Enabling universe repository
[2025-06-20 23:44:45] INFO: Universe enabled
[2025-06-20 23:44:46] INFO: Updating package lists
[2025-06-20 23:44:48] WARNING: Toolchain PPA unavailable, continuing
[2025-06-20 23:44:49] INFO: Package lists updated
[2025-06-20 23:44:50] INFO: Installing build tools
[2025-06-20 23:44:51] INFO: Build tools installed
[2025-06-20 23:44:52] INFO: Installing GCC 8
[2025-06-20 23:44:53] WARNING: GCC-8 install failed
[2025-06-20 23:44:54] INFO: Configuring GCC alternatives
[2025-06-20 23:44:54] INFO: Active GCC: gcc (Ubuntu 14.2.0-4ubuntu2) 14.2.0
[2025-06-20 23:44:55] INFO: Verifying cmake installation
[2025-06-20 23:44:55] INFO: CMake verified
[2025-06-20 23:44:56] INFO: Installing dev libraries
[2025-06-20 23:44:57] INFO: Dev libraries installed
[2025-06-20 23:44:58] INFO: Installing Python dev
[2025-06-20 23:44:58] INFO: Python dev installed
[2025-06-20 23:44:59] INFO: Installing utilities
[2025-06-20 23:45:00] INFO: Utilities installed
[2025-06-20 23:45:01] INFO: System dependencies completed
[2025-06-20 23:45:02] INFO: Setting up llama.cpp
[2025-06-20 23:45:03] INFO: Cloning attempt 1/3
[2025-06-20 23:45:32] INFO: Clone verified successfully
[2025-06-20 23:45:33] INFO: Compiling llama.cpp...
[2025-06-20 23:45:34] INFO: Checking compiler compatibility
[2025-06-20 23:45:35] INFO: CUDA 11.0 max gcc: 8
[2025-06-20 23:45:36] WARNING: gcc-8 not found, checking system GCC
[2025-06-20 23:45:37] INFO: System GCC version: 14
[2025-06-20 23:45:38] WARNING: System gcc-14 too new for CUDA 11.0
[2025-06-20 23:45:38] WARNING: CUDA compilation may fail
[2025-06-20 23:45:39] INFO: Configuring CMake build
[2025-06-20 23:45:41] ERROR: CMake failed: -- The C compiler identification is GNU 14.2.0
...
```

### Requirements
- O.S. - Linux Only (Ubuntu ~24.10 recommended).
- Python - Unknown min version (will be assessed after working version, but presumed 3.8+).
- G.P.U. - nVidia with nVidia driver 450.80+, may be used/compute GPU. 
- Cuda Toolkit - You must install, Version [11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive), depending upon Cuda level of card for processing.
- R.A.M. - Models are stored in Unified Memory, and processes on the GPU. This method should cover most model cases.

### Development
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
