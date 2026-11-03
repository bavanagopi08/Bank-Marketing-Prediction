# Bank Marketing Prediction Using Machine Learning

## Project Overview
This project predicts whether a bank customer will subscribe to a term deposit using supervised machine learning algorithms.

## Internship Task
Build a machine learning model to predict outcomes based on the given dataset.

## Dataset
- Dataset: Bank Marketing Dataset
- Source: UCI Machine Learning Repository
- Task: Binary Classification
- Target Variable: y (Yes / No)

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Model Accuracy Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 83.03% |
| Decision Tree | 84.63% |
| Random Forest | 87.38% |

Best Model: Random Forest

## Model Evaluation
- Confusion Matrix
- ROC Curve
- ROC AUC Score: 0.7851
- Accuracy Comparison Graph

## Project Features
- Data Preprocessing
- Exploratory Data Analysis
- Machine Learning Model Training
- Model Evaluation
- Customer Subscription Prediction
- Streamlit Web Application

## How to Run

1. Clone the repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Run the application:

   streamlit run app.py

## Project Files
- main.py - Model training
- app.py - Prediction web application
- bank_marketing_model.pkl - Trained model
- preprocessor.pkl - Data preprocessing object

## Conclusion
The Random Forest model achieved the highest accuracy of 87.38% among the three tested algorithms.
This project provided practical experience in supervised learning and model evaluation.