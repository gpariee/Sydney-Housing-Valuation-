# Sydney Housing Valuation — Machine Learning Project

Full machine-learning lifecycle case study predicting Sydney house prices across three suburbs
(Mosman, Marrickville, Blacktown): data exploration, feature engineering, model comparison
(Ridge / Random Forest / HistGradientBoosting) with 5-fold cross-validation, prediction-error
analysis, an ML-vs-LLM-vs-human valuation benchmark, and a deployed Flask prediction app.

**⚠️ Data note:** `raw_listings.csv` is an placeholder dataset with real sold-listing records from real estate.com.au and domain.com.au

## Contents
| File | Purpose |
|---|---|
| `Sydney_Housing_Valuation.ipynb` | Full end-to-end notebook (Parts 1–6), With outputs |
| `raw_listings.csv` | Input dataset |
| `eda.py`, `feature_eng.py`, `model.py`, `error_analysis.py` | Equivalent standalone scripts |
| `build_report.py` | Generates the PDF report |
| `app/` | Flask web app serving the trained model |

## Setup
```bash
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
```

## Run the notebook
```bash
jupyter notebook Sydney_Housing_Valuation.ipynb
```

## Run the standalone scripts (same pipeline, script form)
```bash
python eda.py
python feature_eng.py
python model.py
python error_analysis.py
python build_report.py
```

## Run the web app
```bash
cd app
python app.py
# open http://127.0.0.1:5000
```

## Project structure
```
.
├── Sydney_Housing_Valuation.ipynb
├── raw_listings.csv
├── eda.py
├── feature_eng.py
├── model.py
├── error_analysis.py
├── build_report.py
├── requirements.txt
├── app/
│   ├── app.py
│   ├── ridge_model.joblib
│   └── templates/index.html
└── README.md
```
