import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data():
    """Load the Weather Dataset."""
    data = pd.read_csv(r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\data\raw\Project 1 - Weather Dataset.csv")
    print("Dataset loaded successfully. Shape:", data.shape)
    return data

def handle_missing_values(data):
    """Handle missing values in numerical and categorical columns."""
    numerical_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())
    
    data['Weather'] = data['Weather'].fillna(data['Weather'].mode()[0])
    
    print("\nMissing values after imputation:")
    print(data.isnull().sum())
    return data

def convert_data_types(data):
    """Convert Date/Time to datetime and ensure numerical columns are float."""
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    numerical_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
    for col in numerical_cols:
        data[col] = data[col].astype(float)
    
    print("\nData types after conversion:")
    print(data.dtypes)
    return data

def simplify_weather_conditions(data):
    """Simplify Weather column to five broad categories."""
    # Create a copy of the original Weather column for debugging
    data['Weather_Original'] = data['Weather']
    
    weather_mapping = {
        'Clear': 'Clear',
        'Mainly Clear': 'Clear',
        'Cloudy': 'Cloudy',
        'Mostly Cloudy': 'Cloudy',
        'Partly Cloudy': 'Cloudy',
        'Rain': 'Rain',
        'Light Rain': 'Rain',
        'Moderate Rain': 'Rain',
        'Heavy Rain': 'Rain',
        'Rain Showers': 'Rain',
        'Light Rain Showers': 'Rain',
        'Moderate Rain Showers': 'Rain',
        'Heavy Rain Showers': 'Rain',
        'Rain,Fog': 'Rain',
        'Freezing Rain': 'Rain',
        'Freezing Drizzle': 'Rain',
        'Thunderstorms': 'Rain',
        'Thunderstorms,Heavy Rain Showers': 'Rain',
        'Thunderstorms,Moderate Rain Showers,Fog': 'Rain',
        'Thunderstorms,Rain,Fog': 'Rain',
        'Thunderstorms,Rain Showers': 'Rain',
        'Thunderstorms,Rain Showers,Fog': 'Rain',
        'Thunderstorms,Moderate Rain Showers': 'Rain',
        'Thunderstorms,Heavy Rain': 'Rain',
        'Rain,Haze': 'Rain',
        'Rain,Ice Pellets': 'Rain',
        'Rain,Snow Grains': 'Rain',
        'Rain,Snow': 'Rain',
        'Rain,Snow,Ice Pellets': 'Rain',
        'Snow': 'Snow',
        'Light Snow': 'Snow',
        'Moderate Snow': 'Snow',
        'Snow,Fog': 'Snow',
        'Snow Showers': 'Snow',
        'Snow Showers,Fog': 'Snow',
        'Snow Pellets': 'Snow',
        'Freezing Rain,Snow Grains': 'Snow',
        'Freezing Rain,Ice Pellets,Fog': 'Snow',
        'Rain,Snow,Fog': 'Snow',
        'Snow,Blowing Snow': 'Snow',
        'Moderate Snow,Blowing Snow': 'Snow',
        'Snow,Haze': 'Snow',
        'Fog': 'Fog',
        'Mist': 'Fog',
        'Haze': 'Fog',
        'Drizzle': 'Fog',
        'Drizzle,Fog': 'Fog',
        'Drizzle,Ice Pellets,Fog': 'Fog',
        'Drizzle,Snow': 'Fog',
        'Drizzle,Snow,Fog': 'Fog',
        'Freezing Drizzle,Haze': 'Fog',
        'Freezing Drizzle,Snow': 'Fog',
        'Freezing Fog': 'Fog',
        'Freezing Rain,Haze': 'Rain'
    }
    
    data['Weather'] = data['Weather_Original'].map(weather_mapping)
    
    # Check for unmapped conditions
    if data['Weather'].isnull().any():
        print("\nWarning: Unmapped weather conditions found:")
        unmapped_conditions = data[data['Weather'].isnull()]['Weather_Original'].unique()
        print("Unmapped conditions:", list(unmapped_conditions))
        print("Counts of unmapped conditions:")
        print(data[data['Weather'].isnull()]['Weather_Original'].value_counts())
        # Map any remaining conditions to Fog as a fallback
        data['Weather'] = data['Weather'].fillna('Fog')
        print("\nUnmapped conditions mapped to 'Fog' as fallback.")
    
    # Drop the temporary Weather_Original column
    data = data.drop(columns=['Weather_Original'])
    
    print("\nWeather conditions after simplification:")
    print(data['Weather'].value_counts())
    return data

def handle_outliers(data):
    """Cap outliers in numerical features using IQR method."""
    numerical_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
    
    print("\nSummary statistics after outlier handling:")
    print(data[numerical_cols].describe())
    return data

def scale_features(data):
    """Scale numerical features using StandardScaler."""
    numerical_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    print("\nSummary statistics after scaling:")
    print(data[numerical_cols].describe())
    return data

def save_preprocessed_data(data):
    """Save the preprocessed dataset."""
    output_dir = r"C:\Users\Tejas\OneDrive\Desktop\weather_prediction_project\data\processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'processed_data.csv')
    data.to_csv(output_path, index=False)
    print(f"\nPreprocessed data saved to: {output_path}")

def main():
    """Run all preprocessing steps."""
    print("Starting preprocessing for Weather Dataset\n")
    
    data = load_data()
    data = handle_missing_values(data)
    data = convert_data_types(data)
    data = simplify_weather_conditions(data)
    data = handle_outliers(data)
    data = scale_features(data)
    save_preprocessed_data(data)
    
    print("\nPreprocessing complete. Ready for feature engineering.")

if __name__ == "__main__":
    main()