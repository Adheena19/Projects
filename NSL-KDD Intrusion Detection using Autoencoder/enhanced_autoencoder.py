# enhanced_autoencoder.py (enhanced with performance fixes + SVM + Isolation Forest)

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             f1_score, precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras import layers, regularizers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.preprocessor = None

    def load_data(self):
        logger.info("Loading NSL-KDD dataset...")
        try:
            data = load_dataset("Mireu-Lab/NSL-KDD")
            return pd.DataFrame(data['train']), pd.DataFrame(data['test'])
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def preprocess(self, df_train, df_test):
        logger.info("Preprocessing data...")
        categorical_cols = ['protocol_type', 'service', 'flag']
        numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()

        for col in ['src_bytes', 'dst_bytes', 'duration']:
            df_train[col] = np.log1p(df_train[col])
            df_test[col] = np.log1p(df_test[col])

        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        label_col = 'class'
        df_train_normal = df_train[df_train[label_col] == 'normal']
        X_train = self.preprocessor.fit_transform(df_train_normal.drop(columns=[label_col]))
        X_val, X_train = X_train[:100], X_train[100:]  # small val split
        X_test = self.preprocessor.transform(df_test.drop(columns=[label_col]))
        y_test = df_test[label_col].apply(lambda x: 0 if x == 'normal' else 1).values
        return X_train, X_val, X_test, y_test

class EnhancedAutoencoder:
    def __init__(self, input_dim):
        self.model = self.build_model(input_dim)

    def build_model(self, input_dim):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.Dense(128, activation='relu',
                               kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(input_dim, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, X_val):
        return self.model.fit(
            X_train, X_train,
            epochs=50,
            batch_size=256,
            validation_data=(X_val, X_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3)
            ],
            verbose=1
        )

    def reconstruct(self, X):
        return self.model.predict(X)

class AnomalyDetector:
    def __init__(self):
        self.threshold = None

    def calculate_threshold(self, y_val, val_errors):
        prec, rec, thresh = precision_recall_curve(y_val, val_errors)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
        self.threshold = thresh[np.argmax(f1s)]
        logger.info(f"Optimal threshold based on F1: {self.threshold:.6f}")
        return self.threshold

    def evaluate(self, mse, y_test):
        y_pred = (mse > self.threshold).astype(int)
        print("\nAutoencoder Results:")
        print(classification_report(y_test, y_pred))
        print("ROC AUC:", roc_auc_score(y_test, mse))
        return y_pred

if __name__ == "__main__":
    processor = DataProcessor()
    df_train, df_test = processor.load_data()
    X_train, X_val, X_test, y_test = processor.preprocess(df_train, df_test)

    ae = EnhancedAutoencoder(input_dim=X_train.shape[1])
    ae.train(X_train, X_val)

    X_val_pred = ae.reconstruct(X_val)
    val_mse = np.mean(np.square(X_val - X_val_pred), axis=1)

    detector = AnomalyDetector()
    detector.calculate_threshold(np.zeros(len(val_mse)), val_mse)

    X_test_pred = ae.reconstruct(X_test)
    mse_test = np.mean(np.square(X_test - X_test_pred), axis=1)
    detector.evaluate(mse_test, y_test)

    # Comparison 1: One-Class SVM
    ocsvm = OneClassSVM(gamma='auto', nu=0.1)
    ocsvm.fit(X_train)
    svm_pred = ocsvm.predict(X_test)
    svm_pred = np.where(svm_pred == -1, 1, 0)
    print("\nOne-Class SVM:")
    print(classification_report(y_test, svm_pred))

    # Comparison 2: Isolation Forest
    iforest = IsolationForest(contamination=0.1, random_state=42)
    iforest.fit(X_train)
    if_pred = iforest.predict(X_test)
    if_pred = np.where(if_pred == -1, 1, 0)
    print("\nIsolation Forest:")
    print(classification_report(y_test, if_pred))