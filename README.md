# PdM Rubric Scoring (Streamlit)

A Streamlit app to score literature case studies using a shared PdM rubric, with hover tooltips for criteria and a SQLite-backed repository.

## Local setup

```bash
# 1) Create & activate a venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
