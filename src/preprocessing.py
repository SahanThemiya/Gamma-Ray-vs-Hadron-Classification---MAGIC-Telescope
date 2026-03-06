import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

FEATURE_COLS = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1'
    'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist'
]

TARGET_COLS = 'class'
CLASS_MAP = {'g':1, 'h':0}
CLASS_NAMES = ['Hadron', 'Gamma']
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"

def load_data():
    path = os.path.join(DATA_URL, 'magic04.data')
    cols = FEATURE_COLS + TARGET_COLS
    df = pd.read_csv(path, header=None, names=cols)
    df['target'] = df[TARGET_COLS].map(CLASS_MAP)
    return df

def split_data(x, y, val_size=0.15, test_size=0.15, seed=42):
    temp_size = val_size + test_size
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=temp_size, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=test_size/temp_size, random_state=seed, stratify=y_temp
    )
    return x_train,x_val, x_test, y_train, y_val, y_test

def prepare_data(df, feature_cols=None, apply_smote=True, seed=42):
    feature_cols = FEATURE_COLS
    X = df[feature_cols].values
    y = df['target'].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if apply_smote:
        X_train, y_train = SMOTE(random_state=seed).fit_resample(X_train, y_train)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler, 'features': feature_cols,
    }


