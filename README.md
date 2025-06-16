# Chat-Gradio-Linux

### Description
Chat-Gradio-Linux is the Ubuntu version of "Chat-Gradio-Gguf", the plan for the differences are...
- Cuda Only, gpu must still be selectable.
- Installer will only have options for Cuda 11 and Cuda 12.
- The user must install cuda toolkit themselves.
- The scripts MUST use unified memory, so as to do the processing on the selected card, while loading models to memory.
