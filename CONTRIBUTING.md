# Contributing / Setup

Windows quick start:

1. Create virtual environment and activate:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

Or using cmd:

```bat
python -m venv .venv
call .venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train demo artifacts (creates TF-IDF and model in `src/models/artifacts`):

```powershell
python scripts\train_models.py
```

4. Run the dashboard:

```powershell
streamlit run app.py
```

Testing:

```powershell
pip install pytest
pytest -q
```
