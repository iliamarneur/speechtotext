@echo off
chcp 65001 >nul 2>&1
title Transcription Audio - Serveur
echo.
echo ============================================================
echo   TRANSCRIPTION AUDIO - Demarrage du serveur
echo ============================================================
echo.

:: Vérifier que l'installation a été faite
if not exist "venv" (
    echo [ERREUR] L'installation n'a pas ete faite.
    echo Lancez d'abord : install-windows.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo   Modele : medium (bonne qualite, compatible CPU)
echo.
echo   IMPORTANT : Au premier lancement, le modele va se
echo   telecharger (~1.5 Go). Cela peut prendre quelques minutes.
echo.
echo   Une fois pret, ouvrez votre navigateur a l'adresse :
echo.
echo       http://localhost:8000/
echo.
echo   Pour arreter le serveur, fermez cette fenetre.
echo ============================================================
echo.

set WHISPER_MODEL=medium
set WHISPER_DEVICE=auto
set WHISPER_COMPUTE_TYPE=auto
set WHISPER_MODEL_DIR=models

uvicorn main:app --host 0.0.0.0 --port 8000

pause
