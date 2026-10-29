@echo off
cd /d "%~dp0"
echo Starting SLOT HAPPENS Course Finder dashboard...
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
