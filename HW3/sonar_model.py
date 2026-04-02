# ==========================================
# sonar_model.py
# CREATED: 2026
# Project: Sonar Mine Classifier (New Version)
# ==========================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. SETUP: Configure reproducible results
np.random.seed(42)
tf.random.set_seed(42)

def train_new_model():
    print("\n--- NEW Sonar AI Training Sequence Initiated ---")

    # 2. LOAD DATASET (Sonar: Rocks vs. Mines)
    # This is a different data source than your friend's project.
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
        df = pd.read_csv(url, header=None)
        print("Data loaded successfully from UCI Archive.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. PREPROCESSING
    # Split features (X, 60 readings) from labels (y, 'R' or 'M')
    X = df.iloc[:, 0:60].values
    y = df.iloc[:, 60].values

    # Encode labels (R=1, M=0)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split into training and testing sets (New Split Ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

    # Standardize data (Essential for Neural Nets)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Save the scaler so the UI can normalize user data later
    np.save('scaler_mean.npy', sc.mean_)
    np.save('scaler_std.npy', sc.scale_)
    print("Preprocessors (Encoder & Scaler) finalized.")

    # 4. BUILD THE NEW ARCHITECTURE
    # A different configuration of layers and nodes than your friend's model.
    model = Sequential([
        # Input Layer + Hidden Layer 1 (60 inputs, 24 neurons)
        Dense(units=24, activation='relu', input_dim=60),
        # Dropout layer to prevent overfitting (a dynamic feature)
        Dropout(0.2),
        # Hidden Layer 2 (12 neurons)
        Dense(units=12, activation='relu'),
        # Output Layer (1 neuron for Binary Classification: 0-1)
        Dense(units=1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("New model architecture built and compiled.")

    # 5. TRAIN
    # We use a unique training curve (Epochs=35, Batch=8)
    print("Beginning Training (Epochs=35)...")
    history = model.fit(X_train, y_train, batch_size=8, epochs=35, verbose=0)
    print("Training complete.")

    # 6. EVALUATE
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

    # 7. SAVE (This creates the 'brain' file)
    model.save('sonar_classifier.h5')
    print("Saved model brain to 'sonar_classifier.h5'")
    
    # We also save the loss history to visualize in the HTML
    np.save('loss_history.npy', history.history['loss'])
    print("Training loss history exported.")

if __name__ == "__main__":
    train_new_model()