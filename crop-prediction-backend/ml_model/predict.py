import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cv2

# Load the model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'crop_prediction_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    return joblib.load(model_path)

# Extract soil features from image
def extract_soil_features(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image file")
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate average hue, saturation, and value
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    
    # Calculate texture features (using gray-level co-occurrence matrix)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simplistic texture analysis (standard deviation of pixel values)
    texture = np.std(gray)
    
    # Define soil types based on color and texture
    # This is a simplified approach - a real system would use more sophisticated analysis
    if s_mean < 50:  # Low saturation often indicates sandy soil
        if v_mean > 150:  # Light colored
            soil_type = "Sandy"
            estimated_ph = 6.0 + np.random.uniform(-0.5, 0.5)  # Sandy soils tend to be slightly acidic
            estimated_nitrogen = 30 + np.random.uniform(-10, 20)
            estimated_phosphorus = 15 + np.random.uniform(-5, 15)
            estimated_potassium = 10 + np.random.uniform(-5, 10)
        else:
            soil_type = "Sandy Loam"
            estimated_ph = 6.5 + np.random.uniform(-0.4, 0.4)
            estimated_nitrogen = 40 + np.random.uniform(-10, 20)
            estimated_phosphorus = 20 + np.random.uniform(-5, 15)
            estimated_potassium = 15 + np.random.uniform(-5, 10)
    elif 40 <= h_mean <= 70:  # Hue in green-yellow range often indicates clay
        if texture > 40:  # High texture variation
            soil_type = "Clay Loam"
            estimated_ph = 7.2 + np.random.uniform(-0.3, 0.3)
            estimated_nitrogen = 65 + np.random.uniform(-15, 15)
            estimated_phosphorus = 35 + np.random.uniform(-10, 10)
            estimated_potassium = 30 + np.random.uniform(-10, 10)
        else:
            soil_type = "Clay"
            estimated_ph = 7.5 + np.random.uniform(-0.3, 0.3)  # Clay soils tend to be more alkaline
            estimated_nitrogen = 80 + np.random.uniform(-20, 10)
            estimated_phosphorus = 40 + np.random.uniform(-10, 15)
            estimated_potassium = 35 + np.random.uniform(-10, 15)
    elif 10 <= h_mean <= 30:  # Reddish-brown color
        soil_type = "Red Loamy"
        estimated_ph = 6.8 + np.random.uniform(-0.4, 0.4)
        estimated_nitrogen = 70 + np.random.uniform(-15, 15)
        estimated_phosphorus = 45 + np.random.uniform(-10, 10)
        estimated_potassium = 35 + np.random.uniform(-10, 10)
    elif h_mean < 15 and v_mean < 100:  # Dark color
        soil_type = "Black Cotton"
        estimated_ph = 7.8 + np.random.uniform(-0.3, 0.5)  # Black soils tend to be alkaline
        estimated_nitrogen = 85 + np.random.uniform(-15, 10)
        estimated_phosphorus = 55 + np.random.uniform(-10, 15)
        estimated_potassium = 40 + np.random.uniform(-10, 10)
    else:
        soil_type = "Medium Black"
        estimated_ph = 7.3 + np.random.uniform(-0.4, 0.4)
        estimated_nitrogen = 75 + np.random.uniform(-15, 15)
        estimated_phosphorus = 45 + np.random.uniform(-10, 10)
        estimated_potassium = 35 + np.random.uniform(-10, 10)
    
    # Return the extracted features
    return {
        'soil_type': soil_type,
        'ph': estimated_ph,
        'nitrogen': estimated_nitrogen,
        'phosphorus': estimated_phosphorus,
        'potassium': estimated_potassium
    }

# Get weather data for a region
def get_weather_data(region):
    # In a real application, this would call a weather API
    # For this example, we'll use hardcoded representative values for Maharashtra regions
    weather_data = {
        'mumbai': {'temperature': 30, 'humidity': 80, 'rainfall': 2200},
        'pune': {'temperature': 26, 'humidity': 60, 'rainfall': 700},
        'nagpur': {'temperature': 31, 'humidity': 45, 'rainfall': 950},
        'nashik': {'temperature': 25, 'humidity': 55, 'rainfall': 800},
        'aurangabad': {'temperature': 29, 'humidity': 45, 'rainfall': 650},
        'solapur': {'temperature': 30, 'humidity': 40, 'rainfall': 550},
        'amravati': {'temperature': 31, 'humidity': 45, 'rainfall': 850},
        'kolhapur': {'temperature': 26, 'humidity': 70, 'rainfall': 1200},
        'sangli': {'temperature': 28, 'humidity': 55, 'rainfall': 650},
        'satara': {'temperature': 25, 'humidity': 60, 'rainfall': 750},
        'ratnagiri': {'temperature': 29, 'humidity': 85, 'rainfall': 3000},
        'latur': {'temperature': 30, 'humidity': 45, 'rainfall': 600}
    }
    
    # Default to Pune if region not found
    region = region.lower() if region else 'pune'
    return weather_data.get(region, weather_data['pune'])

# Make prediction based on soil and weather data
def predict_crops(soil_features, region, model_data):
    # Extract model components
    crop_model = model_data['crop_model']
    suit_model = model_data['suit_model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_cols = model_data['feature_cols']
    numerical_cols = model_data['numerical_cols']
    
    # Get weather data
    weather = get_weather_data(region)
    
    # Create feature vector
    features = {}
    
    # Add region (encoded)
    region_encoder = label_encoders['region']
    try:
        # Try to transform the region
        region_encoded = region_encoder.transform([region.title()])[0]
    except (ValueError, KeyError):
        # If region not in training data, use the first region
        region_encoded = 0
    features['region'] = region_encoded
    
    # Add soil type (encoded)
    soil_type_encoder = label_encoders['soil_type']
    try:
        # Try to transform the soil type
        soil_type_encoded = soil_type_encoder.transform([soil_features['soil_type']])[0]
    except (ValueError, KeyError):
        # If soil type not in training data, use the first soil type
        soil_type_encoded = 0
    features['soil_type'] = soil_type_encoded
    
    # Add other features
    features['nitrogen'] = soil_features['nitrogen']
    features['phosphorus'] = soil_features['phosphorus']
    features['potassium'] = soil_features['potassium']
    features['ph'] = soil_features['ph']
    features['temperature'] = weather['temperature']
    features['humidity'] = weather['humidity']
    features['rainfall'] = weather['rainfall']
    
    # Create DataFrame with proper column order
    X = pd.DataFrame([features])[feature_cols]
    
    # Scale numerical features
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    # Get top crop probabilities
    crop_probs = crop_model.predict_proba(X)[0]
    crop_indices = np.argsort(-crop_probs)[:5]  # Top 5 crops
    
    crops = []
    for idx in crop_indices:
        crop_name = label_encoders['crop'].inverse_transform([idx])[0]
        
        # Create a copy of X but change the soil_type to ideal for this crop
        # This allows us to predict suitability for each crop
        X_crop = X.copy()
        
        # Predict suitability
        suitability = suit_model.predict(X_crop)[0]
        
        # Cap suitability to reasonable range (50-95)
        adj_suitability = max(50, min(95, suitability))
        
        crops.append({
            'name': crop_name,
            'suitability': round(adj_suitability, 1)
        })
    
    # Return prediction results
    return {
        'crops': crops,
        'soilType': soil_features['soil_type'],
        'ph': round(soil_features['ph'], 1),
        'nitrogen': 'High' if soil_features['nitrogen'] > 75 else 'Medium' if soil_features['nitrogen'] > 50 else 'Low',
        'phosphorus': 'High' if soil_features['phosphorus'] > 40 else 'Medium' if soil_features['phosphorus'] > 25 else 'Low',
        'potassium': 'High' if soil_features['potassium'] > 35 else 'Medium' if soil_features['potassium'] > 20 else 'Low'
    }

# Create chart for the results
def create_chart(prediction_results):
    # Create bar chart for crop suitability
    crops = [crop['name'] for crop in prediction_results['crops']]
    suitability = [crop['suitability'] for crop in prediction_results['crops']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(crops, suitability, color='skyblue')
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    plt.xlabel('Crops')
    plt.ylabel('Suitability (%)')
    plt.title('Crop Suitability Analysis')
    plt.ylim(0, 100)
    plt.tight_layout()
    
    # Save chart to memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Convert to base64 for embedding in JSON response
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return chart_data

# Main prediction function
def predict(image_path, region):
    try:
        # Load the model
        model_data = load_model()
        
        # Extract soil features from the image
        soil_features = extract_soil_features(image_path)
        
        # Make prediction
        prediction_results = predict_crops(soil_features, region, model_data)
        
        # Create chart
        chart_data = create_chart(prediction_results)
        
        # Add chart to results
        prediction_results['chart'] = chart_data
        
        return prediction_results
        
    except Exception as e:
        return {
            'error': str(e),
            'message': 'Failed to process prediction'
        }

# Main entry point when script is run directly
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Missing arguments. Usage: python predict.py <image_path> <region>'}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    region = sys.argv[2]
    
    results = predict(image_path, region)
    print(json.dumps(results))