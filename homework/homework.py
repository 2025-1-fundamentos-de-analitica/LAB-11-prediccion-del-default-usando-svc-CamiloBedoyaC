import zipfile
import pickle
import gzip
import json
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def clean_data(df):
    df = df.copy()
    df = df.drop('ID', axis=1)
    df = df.rename(columns={'default payment next month': 'default'})
    df = df.dropna()
    df = df[(df['EDUCATION'] != 0 ) & (df['MARRIAGE'] != 0)]
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df

def model():
    categories = ['SEX', 'EDUCATION', 'MARRIAGE']  
    numerics = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categories),
            ('scaler', StandardScaler(), numerics)
        ],
        remainder='passthrough'
    )
    selectkbest = SelectKBest(score_func=f_classif)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA()),
        ('selectkbest', selectkbest),
        ('classifier', SVC(kernel='rbf', random_state=42))
    ])
    return pipeline

def hyperparameters(model, n_splits, x_train, y_train, scoring):
    estimator = GridSearchCV(
        estimator=model,
        param_grid = {
            'pca__n_components': [20, 21],  
            'selectkbest__k': [12],
            'classifier__kernel': ['rbf'],
            'classifier__gamma': [0.099]
        },
        cv=n_splits,
        refit=True,
        verbose=1,
        return_train_score=False,
        scoring=scoring
    )
    estimator.fit(x_train, y_train)
    return estimator

def metrics(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred)
    }
    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred)
    }
    return train_metrics, test_metrics

def matrix(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    cm_train = confusion_matrix(y_train, y_train_pred).ravel()
    cm_test = confusion_matrix(y_test, y_test_pred).ravel()
    train_matrix = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {'predicted_0': int(cm_train[0]), 'predicted_1': int(cm_train[1])},
        'true_1': {'predicted_0': int(cm_train[2]), 'predicted_1': int(cm_train[3])}
    }
    test_matrix = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {'predicted_0': int(cm_test[0]), 'predicted_1': int(cm_test[1])},
        'true_1': {'predicted_0': int(cm_test[2]), 'predicted_1': int(cm_test[3])}
    }
    return train_matrix, test_matrix

def save_model(model):
    os.makedirs('files/models', exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)

def save_metrics(metrics):
    os.makedirs('files/output', exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            json_line = json.dumps(metric)
            f.write(json_line + "\n")

# Main execution
if __name__ == "__main__":
    file_Test = 'files/input/test_data.csv.zip'
    file_Train = 'files/input/train_data.csv.zip'
    with zipfile.ZipFile(file_Test, 'r') as zipf:
        with zipf.open('test_default_of_credit_card_clients.csv') as f:
            df_Test = pd.read_csv(f)
    with zipfile.ZipFile(file_Train, 'r') as zipf:
        with zipf.open('train_default_of_credit_card_clients.csv') as f:
            df_Train = pd.read_csv(f)
    df_Test = clean_data(df_Test)
    df_Train = clean_data(df_Train)
    x_train, y_train = df_Train.drop('default', axis=1), df_Train['default']
    x_test, y_test = df_Test.drop('default', axis=1), df_Test['default']
    model_pipeline = model()
    model_pipeline = hyperparameters(model_pipeline, 10, x_train, y_train, 'balanced_accuracy')
    save_model(model_pipeline)
    train_metrics, test_metrics = metrics(model_pipeline, x_train, y_train, x_test, y_test)
    train_matrix, test_matrix = matrix(model_pipeline, x_train, y_train, x_test, y_test)
    save_metrics([train_metrics, test_metrics, train_matrix, test_matrix])