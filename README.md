# Chat-Linux-Gguf

### Description
Chat-Linux-Gguf is the Ubuntu version of [Chat-Gradio-Gguf](https://github.com/wiseman-timelord/Chat-Gradio-Gguf) which runs on Windows. So, the plan for the differences are...
- .sh instead of .bat, as well as program the scripts for specifically linux/ubuntu.
- Cuda Only, gpu must still be selectable. Installer will only have options for Cuda 11 and Cuda 12.
- The user must install cuda toolkit themselves.
- The scripts MUST use unified memory, so as to do the processing on the selected card, while loading models to system memory. There will be no requirement for calculating how many layers to load to the GPU, we will assume that the models will load on the system memory available. optimizing functions/processes that were simplified for removal of the calculations.

### Notation
- In-short Ubuntu 24.10 has allowed me to install, "RX 470" as main with "1060 3GB" as compute for cuda, with no issues, thanks to help from [X-Grok](www.x.com).
