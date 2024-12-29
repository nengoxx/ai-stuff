@echo off
REM Navigate to the directory where this script is located
cd %~dp0

REM Create a virtual environment
set VENV_FOLDER=.venv
set PYTHON_VERSION=3.12

if not exist %VENV_FOLDER% (
    py -%PYTHON_VERSION% -m venv %VENV_FOLDER%
)

call %VENV_FOLDER%\Scripts\activate


REM This command will install the project in editable mode along with all its required dependencies. You can specify the extras (inside the pyproject.toml)
pip install uv
uv pip install -e ".[client,dev,ui,opentelemetry]"


REM Set environment variables
set WHISPER__MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
set DEFAULT_LANGUAGE=en
set UVICORN_HOST=0.0.0.0
set UVICORN_PORT=9000

REM Run the application
uvicorn faster_whisper_server.main:create_app --host %UVICORN_HOST% --port %UVICORN_PORT%