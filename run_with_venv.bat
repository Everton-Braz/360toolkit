@echo off
REM run_with_venv.bat — Activates repo venv (if present) and runs run_app.py
setlocal enabledelayedexpansion
set "REPO_DIR=%~dp0"
set "FOUND_VENV="
for %%v in (.venv venv env .env) do (
  if exist "%REPO_DIR%%%v\Scripts\activate.bat" (
    set "FOUND_VENV=%REPO_DIR%%%v"
    goto :found
  )
)
:found
if "%FOUND_VENV%"=="" (
  echo No virtual environment found in .venv, venv, env, or .env.
  echo To create one: python -m venv .venv
  pause
  exit /b 1
)
call "%FOUND_VENV%\Scripts\activate.bat"
echo Activated %FOUND_VENV%
python "%REPO_DIR%run_app.py" %*
endlocal
exit /b %ERRORLEVEL%
