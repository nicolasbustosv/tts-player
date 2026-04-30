@echo off
title TTS Player
call "%USERPROFILE%\AppData\Local\anaconda3\Scripts\activate.bat" base
python "%~dp0read.py" --clipboard --speed 5
