# Installing faster-whisper-server directly into windows

We will be installing this project into a virtual environment to be able to edit the source code and enable the Voice Activity Detection feature.

## Installation script (no Docker)

Clone the repo and copy the script inside the root folder. The installation script creates a virtual environment, installs uv, the pyproject.toml and runs the app with uvicorn.

## CuDNN issues

```bash
Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
```

In order to fix this, you will need to install cudnn 9 for cuda 12. **If it doesn't recognize the path, try to copy the files(.dll) inside /bin folder from the CuDNN install (C:\Program Files\NVIDIA\CUDNN\v9.6\bin\12.6) into the /bin folder of your cuda install(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6).**

## Edit faster-whisper-server\src\faster_whisper_server\routers\stt.py to enable VAD by default

Replace the lines `vad_filter: Annotated[bool, Form()] = False,` with True.

Additionally, you can add `beam_size=5,` to the transcribe function calls on that same file to enhance the quality of the transcription.
