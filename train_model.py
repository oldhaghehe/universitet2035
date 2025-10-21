
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def build_mlp_model(input_shape, num_classes=1):
    """
    Builds a Multi-Layer Perceptron (MLP) model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='sigmoid') # Sigmoid for binary classification
    ])
    return model

def compile_model(model):
    """
    Compiles the Keras model with appropriate optimizer, loss, and metrics.
    """
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 
                   keras.metrics.Precision(name='precision'),
                   keras.metrics.Recall(name='recall'),
                   keras.metrics.AUC(name='auc')]
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains the Keras model with EarlyStopping callback.
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set and prints various metrics.
    """
    print("\n--- Model Evaluation ---")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, precision, recall, f1, roc_auc, cm

def save_model(model, path):
    """
    Saves the trained Keras model.
    """
    model.save(path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    # This block is for testing the functions directly
    # In the final pipeline, orchestration will be from the notebook
    print("This script defines model building and training functions. Run from notebook for full pipeline.")
