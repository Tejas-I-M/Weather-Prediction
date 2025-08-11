import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the feature-engineered dataset."""
    data = pd.read_csv(r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\data\processed\engineered_data.csv")
    print("Feature-engineered dataset loaded. Shape:", data.shape)
    return data

def prepare_data(data):
    """Prepare features (X) and target (y) for evaluation."""
    feature_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 
                    'Visibility_km', 'Press_kPa', 'Hour', 'Month', 'DayOfWeek', 
                    'Is_Night', 'Temp_Diff', 'Hum_Vis_Interaction']
    
    # Ensure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    X = data[feature_cols]
    y = data['Weather']
    
    print("\nFeature columns:", X.columns.tolist())
    print("Target column: Weather")
    return X, y

def load_model():
    """Load the trained model."""
    model_path = r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\models\trained_model.pkl"
    model = joblib.load(model_path)
    print(f"\nTrained model loaded from: {model_path}")
    return model

def evaluate_model(model, X, y):
    """Evaluate the model on the test set."""
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\nTest Set Accuracy:", accuracy)
    print("Test Set F1-Score (Macro):", f1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    output_dir = r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()
    print(f"\nConfusion matrix saved to: {output_path}")
    
    return accuracy, f1

def main():
    """Run all evaluation steps."""
    print("Starting model evaluation for Weather Dataset\n")
    
    # Load data
    data = load_data()
    
    # Prepare features and target
    X, y = prepare_data(data)
    
    # Load model
    model = load_model()
    
    # Evaluate model
    accuracy, f1 = evaluate_model(model, X, y)
    
    print("\nModel evaluation complete.")

if __name__ == "__main__":
    main()