@echo off
chcp 65001 >nul 2>&1
title Transcription Audio - Installation
echo.
echo ============================================================
echo   INSTALLATION - Transcription Audio (Whisper)
echo ============================================================
echo.

:: Vérifier Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe.
    echo.
    echo Telechargez Python ici : https://www.python.org/downloads/
    echo IMPORTANT : cochez "Add Python to PATH" pendant l'installation.
    echo.
    pause
    exit /b 1
)
echo [OK] Python detecte
python --version

:: Vérifier ffmpeg
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] ffmpeg n'est pas installe. Installation via winget...
    winget install ffmpeg >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ATTENTION] Impossible d'installer ffmpeg automatiquement.
        echo Telechargez-le ici : https://ffmpeg.org/download.html
        echo Puis relancez ce script.
        pause
        exit /b 1
    )
    echo [OK] ffmpeg installe
) else (
    echo [OK] ffmpeg detecte
)

:: Créer le virtualenv
if not exist "venv" (
    echo.
    echo [...] Creation de l'environnement virtuel...
    python -m venv venv
    echo [OK] Environnement virtuel cree
) else (
    echo [OK] Environnement virtuel existe deja
)

:: Installer les dépendances
echo.
echo [...] Installation des dependances (peut prendre quelques minutes)...
call venv\Scripts\activate.bat
pip install -r requirements.txt >nul 2>&1
pip install requests >nul 2>&1
echo [OK] Dependances installees

echo.
echo ============================================================
echo   INSTALLATION TERMINEE !
echo.
echo   Pour lancer le serveur, double-cliquez sur :
echo     demarrer.bat
echo ============================================================
echo.
pause
