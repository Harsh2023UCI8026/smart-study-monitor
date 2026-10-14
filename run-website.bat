@echo off
setlocal
cd /d "%~dp0web"
echo Local server is starting. Open http://localhost:8000 in Chrome or Edge.
"..\.venv\Scripts\python.exe" -m http.server 8000
endlocal
