[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/6KpniKiX)
# CS506 Midterm

Please take a look at the starter code in the jupyter notebook.

# Amazon Review Rating Prediction (CS506 Midterm)

## Setup
- Python 3.x
- Install dependencies:

pip install -r requirements.txt

## How to run
Run the notebook:

jupyter notebook notebooks/modeling.ipynb

## Model
- TF-IDF (1,2 ngrams, 10k features)
- Ridge Regression

## Result
Validation RMSE: ~1.18

## Notes
- Used text + numeric features
- Test file does not contain features → dummy submission used



features.py 
	•	keeps numeric columns
	•	converts review text into TF-IDF features
	•	combines both into one feature matrix 
    •   assumes  review text column is called Text  text_column="Text" 



Good choices:
	•	TF-IDF for text
	•	Ridge for sparse high-dimensional features
	•	RMSE for evaluation
	•	clip predictions to 1–5 so impossible ratings do not hurt you

For RMSE competitions, it is often better to submit continuous predictions instead of rounding to integers. That is why I set:
round_predictions=False

If you want to try rounded predictions too, change: round_predictions=True

Implemented TF-IDF + Ridge regression model + pipeline and submission flow
