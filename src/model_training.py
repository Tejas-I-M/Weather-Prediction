import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os

def load_data():
    """Load the feature-engineered dataset."""
    data = pd.read_csv(r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\data\processed\engineered_data.csv")
    print("Feature-engineered dataset loaded. Shape:", data.shape)
    return data

def check_rare_classes(data):
    """Check for rare classes in the Weather column (fewer than 2 occurrences)."""
    weather_counts = data['Weather'].value_counts()
    rare_classes = weather_counts[weather_counts < 2].index.tolist()
    if rare_classes:
        print("\nWarning: Rare classes (fewer than 2 occurrences) found in Weather column:")
        print(weather_counts[weather_counts < 2])
        print("Consider revisiting preprocessing.py to group or remove these classes.")
    else:
        print("\nNo rare classes (fewer than 2 occurrences) found in Weather column.")
    return data

def prepare_data(data):
    """Prepare features (X) and target (y) for training."""
    # Define feature columns (exclude Date/Time, Weather, and categorical labels)
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

def train_model(X, y):
    """Train a Random Forest Classifier with hyperparameter tuning."""
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define Random Forest model
    rf = RandomForestClassifier(random_state=42)
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\nBest Hyperparameters:", grid_search.best_params_)
    print("Test Set Accuracy:", accuracy)
    print("Test Set F1-Score (Macro):", f1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print feature importance
    feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    return best_model, X_test, y_test

def save_model(model):
    """Save the trained model."""
    output_dir = r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'trained_model.pkl')
    joblib.dump(model, output_path)
    print(f"\nTrained model saved to: {output_path}")

def main():
    """Run all model training steps."""
    print("Starting model training for Weather Dataset\n")
    
    # Load data
    data = load_data()
    
    # Check for rare classes
    data = check_rare_classes(data)
    
    # Prepare features and target
    X, y = prepare_data(data)
    
    # Train model
    best_model, X_test, y_test = train_model(X, y)
    
    # Save model
    save_model(best_model)
    
    print("\nModel training complete. Ready for evaluation.")

if __name__ == "__main__":
    main()