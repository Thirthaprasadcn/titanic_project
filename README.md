# Titanic Survival Prediction (Streamlit)

A simple web app to predict Titanic passenger survival probability using a pre-trained Random Forest model.

## Project structure

- `app.py`: Streamlit application interface with input widgets and prediction logic.
- `titanic_rf_model.pkl`: Trained RandomForest classifier (Titanic features).
- `test_app.py`: Minimal Streamlit test app.
- `README.md`: Project documentation.

## Requirements

- Python 3.8+
- streamlit
- scikit-learn
- pandas
- numpy
- joblib

## Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

## Run app

```bash
cd titanic_app
python -m streamlit run app.py --server.port 8505
```

Open the URL shown in terminal (likely `http://localhost:8505`).

## Usage

1. Choose passenger details (class, sex, age, family aboard, fare, embarked port).
2. Click `🔍 Predict`.
3. The app displays survival status, probabilities, and progress bars.

## Notes

- If the page is blank, check that Streamlit is running on the correct port and clear browser cache.
- The model was trained on the Titanic dataset (seaborn or Kaggle subset).
