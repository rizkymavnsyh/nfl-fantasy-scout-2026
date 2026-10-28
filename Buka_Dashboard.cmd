@echo off
cd /d "%~dp0"
echo Buka http://localhost:8766 di browser. Tutup jendela ini untuk menghentikan server.
py -3.11 -m http.server 8766 --bind 127.0.0.1 --directory dist
pause
