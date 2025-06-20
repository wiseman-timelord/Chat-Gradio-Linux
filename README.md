# Chat-Linux-Gguf
Status: Alpha - Conversion mostly complete, but not testing yet.

### Description
Chat-Linux-Gguf is the Ubuntu version of [Chat-Gradio-Gguf](https://github.com/wiseman-timelord/Chat-Gradio-Gguf) (which runs on Windows). So, the plan for the differences are...
- .sh instead of .bat, as well as program the scripts for specifically linux/ubuntu.
- Cuda Only, gpu must still be selectable. Installer will only have options for Cuda 11 and Cuda 12. There will also be some optimization/cleanup after
- The user must install cuda toolkit themselves.
- The scripts MUST use unified memory, so as to do the processing on the selected card, while loading models to system memory. There will be no requirement for calculating how many layers to load to the GPU, we will assume that the models will load on the system memory available. optimize/cleanup functions/processes for removal of the calculations.

### Requirements
- O.S. - Linux Only (Ubuntu ~24.10 recommended).
- Python - Unknown min version (will be assessed after working version, but presumed 3.8+).
- G.P.U. - nVidia with nVidia driver 450.80+, may be used/compute GPU. 
- Cuda Toolkit - You must install, Version [11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive), depending upon Cuda level of card for processing.
- R.A.M. - Models are stored in Unified Memory, and processes on the GPU. This method should cover most model cases.



### Development
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
