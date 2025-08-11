import pandas as pd
import numpy as np
import os

def load_data():
    """Load the preprocessed dataset."""
    data = pd.read_csv(r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\data\processed\processed_data.csv")
    print("Preprocessed dataset loaded. Shape:", data.shape)
    return data

def extract_time_features(data):
    """Extract time-based features from Date/Time."""
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    data['Hour'] = data['Date/Time'].dt.hour
    data['Month'] = data['Date/Time'].dt.month
    data['DayOfWeek'] = data['Date/Time'].dt.dayofweek
    
    print("\nTime-based features added: Hour, Month, DayOfWeek")
    return data

def add_is_night(data):
    """Add binary feature indicating nighttime (6 PM to 6 AM)."""
    data['Is_Night'] = ((data['Hour'] >= 18) | (data['Hour'] < 6)).astype(int)
    print("Night indicator feature added: Is_Night")
    return data

def add_temperature_difference(data):
    """Add feature for temperature difference (Temp_C - Dew Point Temp_C)."""
    data['Temp_Diff'] = data['Temp_C'] - data['Dew Point Temp_C']
    print("Temperature difference feature added: Temp_Diff")
    return data

def add_interaction_features(data):
    """Add interaction feature between humidity and visibility."""
    data['Hum_Vis_Interaction'] = data['Rel Hum_%'] * data['Visibility_km']
    print("Interaction feature added: Hum_Vis_Interaction")
    return data

def add_categorical_features(data):
    """Add categorical features for visibility and wind speed."""
    # Visibility categories
    data['Visibility_Category'] = pd.cut(data['Visibility_km'], 
                                        bins=[-float('inf'), 10, 25, float('inf')],
                                        labels=['Low', 'Medium', 'High'])
    
    # Wind speed categories
    data['Wind_Speed_Category'] = pd.cut(data['Wind Speed_km/h'], 
                                         bins=[-float('inf'), 20, 40, float('inf')],
                                         labels=['Low', 'Medium', 'High'])
    
    print("Categorical features added: Visibility_Category, Wind_Speed_Category")
    return data

def save_engineered_data(data):
    """Save the feature-engineered dataset."""
    output_dir = r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\data\processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'engineered_data.csv')
    data.to_csv(output_path, index=False)
    print(f"\nFeature-engineered data saved to: {output_path}")

def main():
    """Run all feature engineering steps."""
    print("Starting feature engineering for Weather Dataset\n")
    
    data = load_data()
    data = extract_time_features(data)
    data = add_is_night(data)
    data = add_temperature_difference(data)
    data = add_interaction_features(data)
    data = add_categorical_features(data)
    save_engineered_data(data)
    
    print("\nFeature engineering complete. Ready for model training.")

if __name__ == "__main__":
    main()