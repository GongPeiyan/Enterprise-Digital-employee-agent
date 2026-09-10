@echo off
title Reset Digital Employee Password
cd /d "%~dp0"
".venv\Scripts\python.exe" "reset_pwd.py"
pause
