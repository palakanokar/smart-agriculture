# Smart Agriculture Monitoring – AI-Based Crop Recommendation System

A Flask web app that recommends the best crop to grow from soil nutrients (N, P, K, pH) and
weather (temperature, humidity, rainfall) using a Scikit-learn **Random Forest Classifier**.

## Project structure
```
smart-agriculture/
├── app.py              # Flask backend (routes /, /predict, /metrics)
├── train_model.py      # Training + evaluation (accuracy, confusion matrix, report)
├── data_loader.py      # Dataset loading & preprocessing (Pandas)
├── explainer.py        # AI explanation (rule-based, optional free Gemini API)
├── dataset/Crop_recommendation.csv
├── model/              # Created by train_model.py (crop_model.pkl, metrics.json)
├── templates/index.html
├── static/style.css, static/script.js
├── requirements.txt
└── README.md
```

## Dataset
- Name: Crop Recommendation Dataset (Atharva Ingle), 2,200 rows, 22 crops, 7 features.
- Source: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- Already included in `dataset/`. If missing: download from Kaggle, unzip, and place
  `Crop_recommendation.csv` inside the `dataset/` folder.
- Columns: `N, P, K, temperature, humidity, ph, rainfall, label`

## Install & run locally
Requires Python 3.9+.
```bash
cd smart-agriculture
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train_model.py        # trains model, prints accuracy, report & confusion matrix
python app.py                # open http://127.0.0.1:5000
```
(`app.py` also trains automatically if the model file is missing.)

### Optional generative AI explanation
The app works fully offline with a rule-based explanation. To use Google Gemini's free tier:
`export GEMINI_API_KEY=your_key` (Windows: `set GEMINI_API_KEY=your_key`) before `python app.py`.

## Methodology flowchart
```
 Dataset (CSV, 2200 rows)
        │
        ▼
 Preprocessing (drop nulls/duplicates, select features)
        │
        ▼
 Train/Test Split (80% / 20%, stratified)
        │
        ▼
 Random Forest Classifier (100 trees) ──► Evaluation (accuracy, confusion matrix, report)
        │
        ▼
 User Input (N, P, K, temp, humidity, pH, rainfall) via web form
        │
        ▼
 Crop Prediction (Flask /predict)
        │
        ▼
 AI Explanation (rule-based or Gemini)
```

## Project explanation (for report)
**Problem:** Farmers often choose crops without scientific analysis of soil and climate, leading to low yield.
**Solution:** A machine-learning system that learns from 2,200 labelled field records which conditions suit
each of 22 crops. The Random Forest algorithm builds many decision trees on random subsets of the data and
features; the final prediction is the majority vote, which reduces overfitting and gives high accuracy.
**Workflow:** The dataset is loaded with Pandas, cleaned, and split 80/20. The model is trained and evaluated
on unseen test data; real accuracy, precision, recall, F1-score and the confusion matrix are saved to
`metrics.json` and shown on the website. Users enter soil and weather values in a validated form; Flask
passes them to the trained model and returns the crop, the model's confidence, and an explanation that
compares the inputs with the crop's typical growing conditions.
**Result:** The model achieves high test accuracy (see the value printed by `train_model.py` / shown on the
website). Rainfall, humidity and potassium are typically the most important features.
**Future scope:** Live IoT sensor input, weather API integration, fertilizer recommendation, regional languages.

## Viva quick notes
- Why Random Forest? Handles non-linear data, robust to outliers, gives feature importance, little tuning.
- Why stratified split? Keeps the same proportion of every crop in train and test sets.
- Confusion matrix: rows = actual crop, columns = predicted; diagonal = correct predictions.

## Deploy online (free – Render.com)
1. Push this folder to a GitHub repository (include `dataset/`; `model/` is regenerated automatically).
2. On https://render.com → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt && python train_model.py`
4. Start command: `gunicorn app:app`
5. Deploy and open the given URL. (PythonAnywhere also works: upload files, create a Flask web app pointing to `app.py`.)
