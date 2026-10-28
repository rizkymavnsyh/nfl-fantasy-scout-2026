@echo off
cd /d "%~dp0"
echo Memperbarui data dan prediksi laga berikutnya. Proses dapat memerlukan beberapa menit.
py -3.11 src\pregame_features.py --refresh-current
if errorlevel 1 goto gagal
py -3.11 src\refresh_forecasts.py
if errorlevel 1 goto gagal
py -3.11 src\test_forecast_dates.py
if errorlevel 1 goto gagal
py -3.11 src\pregame_forecasts.py
if errorlevel 1 goto gagal
py -3.11 src\availability.py
if errorlevel 1 goto gagal
py -3.11 src\export_web_report.py
if errorlevel 1 goto gagal
py -3.11 src\prospective.py --freeze
if errorlevel 1 goto gagal
py -3.11 src\build_notebook.py
if errorlevel 1 goto gagal
py -3.11 src\execute_notebook.py
if errorlevel 1 goto gagal
py -3.11 src\package_project.py
if errorlevel 1 goto gagal
echo Selesai. Buka kembali NFL_Fantasy_Scout_2026.html untuk melihat data terbaru.
pause
exit /b 0
:gagal
echo Pembaruan belum selesai. Periksa pesan di atas. Jangan gunakan hasil ini sebagai data terbaru.
pause
exit /b 1
