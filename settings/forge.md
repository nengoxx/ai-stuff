# Settings for both A1111’s & Forge web UI

**Output Path:** %UserProfile%\\Documents\\sd

**Optimizations:**
\-sdp
\-xFormers (1-3s faster)
\-Enable quantization in K samplers for sharper and cleaner results (no downsides apparently)

## Args:


set SD\_DIR=%UserProfile%\\Documents\\sd
set MODELS\_DIR=%UserProfile%\\Documents\\sd\\models

set DIR\_ARGS=^
\--ckpt-dir "%MODELS\_DIR%\\Stable-diffusion" ^
\--lora-dir "%MODELS\_DIR%\\Lora" ^
\--hypernetwork-dir "%MODELS\_DIR%\\hypernetworks" ^
\--embeddings-dir "%MODELS\_DIR%\\embeddings" ^
\--textual-inversion-templates-dir "%MODELS\_DIR%\\textual\_inversion\_templates" ^
\--vae-dir "%MODELS\_DIR%\\VAE" ^
\--esrgan-models-path "%MODELS\_DIR%\\ESRGAN" ^
\--realesrgan-models-path "%MODELS\_DIR%\\RealESRGAN" ^
\--codeformer-models-path "%MODELS\_DIR%\\Codeformer" ^
\--gfpgan-dir "%MODELS\_DIR%\\GFPGAN" ^
\--gfpgan-models-path "%MODELS\_DIR%\\GFPGAN" ^
\--bsrgan-models-path "%MODELS\_DIR%\\BSRGAN" ^
\--scunet-models-path "%MODELS\_DIR%\\ScuNET" ^
\--swinir-models-path "%MODELS\_DIR%\\SwinIR" ^
\--ldsr-models-path "%MODELS\_DIR%\\LDSR" ^
\--dat-models-path "%MODELS\_DIR%\\DAT" ^
\--clip-models-path "%MODELS\_DIR%\\CLIP"

set COMMANDLINE\_ARGS=%DIR\_ARGS% ^
\--theme dark ^
\--api ^
\--listen ^
\--enable-insecure-extension-access

@REM Uncomment following line for Forge & ReForge.
@REM \--pin-shared-memory \--cuda-malloc \--cuda-stream
@REM Uncomment following code to reference an existing A1111 checkout.
@REM set A1111\_HOME=Your A1111 checkout dir
@REM
@REM set VENV\_DIR=%A1111\_HOME%/venv
@REM set COMMANDLINE\_ARGS=%COMMANDLINE\_ARGS% ^
@REM  \--ckpt-dir %A1111\_HOME%/models/Stable-diffusion ^
@REM  \--hypernetwork-dir %A1111\_HOME%/models/hypernetworks ^
@REM  \--embeddings-dir %A1111\_HOME%/embeddings ^
@REM  \--lora-dir %A1111\_HOME%/models/Lora

