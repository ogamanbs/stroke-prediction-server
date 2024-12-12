from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('./stroke-data.csv')
    df = df.drop('id', axis=1)
    categorical = ['hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical = ['avg_glucose_level', 'bmi', 'age']
    y = df['stroke']
    X = df.drop('stroke', axis=1)
    return X, y, categorical, numerical

def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    return scores

# Load data
X, y, categorical, numerical = load_data()
print(X.shape, y.shape)

# Define the LDA model
model = LinearDiscriminantAnalysis()

# Prepare the pipeline
transformer = ColumnTransformer(transformers=[
    ('imp', SimpleImputer(strategy='median'), numerical),
    ('o', OneHotEncoder(handle_unknown='ignore'), categorical)  # handle_unknown='ignore' to manage unseen categories
])

pipeline = Pipeline(steps=[
    ('t', transformer),
    ('p', PowerTransformer(method='yeo-johnson', standardize=True)),
    ('over', SMOTE()),
    ('m', model)
])

# Evaluate the model
scores = evaluate_model(X, y, pipeline)
# print('LDA %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Plot the results
plt.boxplot([scores], labels=['LDA'], showmeans=True)
plt.show()

# Fit the pipeline on the entire dataset
pipeline.fit(X, y)

# Save the trained pipeline
dump(pipeline, 'stroke_prediction_model.joblib')