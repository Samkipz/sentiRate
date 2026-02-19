python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\scripts\train_models.py
streamlit run .\app.py
