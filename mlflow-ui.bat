@echo off
echo ========================================
echo   LPAK MLOps - MLflow UI
echo ========================================
echo.
echo Dang khoi dong MLflow UI...
echo Truy cap: http://localhost:5001
echo.
mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri file:./mlruns --default-artifact-root ./mlruns
pause



