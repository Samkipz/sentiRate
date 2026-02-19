@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python scripts\train_models.py
streamlit run app.py
pause
