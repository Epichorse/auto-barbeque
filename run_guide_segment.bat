@echo off
setlocal

REM Run guide-based WhisperX segmentation in conda env "whisper".
REM Usage:
REM   run_guide_segment.bat "D:\video\demo.mp4"
REM   run_guide_segment.bat --from-json "output_whisperx\demo.aligned.json"

conda run -n whisper python "%~dp0run_whisperx_guide_dp.py" %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo [error] run_whisperx_guide_dp.py failed with code %EXIT_CODE%
)

exit /b %EXIT_CODE%
