import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

def train_model():
    # Check if model directory exists, if not create it
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'maharashtra_crop_data.csv')
    data = pd.read_csv(dataset_path)
    
    # Preprocess data
    # Encode categorical variables
    label_encoders = {}
    for column in ['region', 'soil_type', 'crop']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    # Split features and target
    X = data.drop(['crop', 'suitability'], axis=1)
    y_crop = data['crop']
    y_suit = data['suitability']
    
    # Split the data
    X_train, X_test, y_crop_train, y_crop_test, y_suit_train, y_suit_test = train_test_split(
        X, y_crop, y_suit, test_size=0.2, random_state=42
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'temperature', 'humidity', 'rainfall']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train Random Forest models
    # 1. Crop Classification model
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X_train, y_crop_train)
    
    # 2. Suitability prediction model
    suit_model = RandomForestRegressor(n_estimators=100, random_state=42)
    suit_model.fit(X_train, y_suit_train)
    
    # Evaluate models
    # Crop classification evaluation
    y_crop_pred = crop_model.predict(X_test)
    crop_accuracy = accuracy_score(y_crop_test, y_crop_pred)
    
    # Suitability prediction evaluation
    y_suit_pred = suit_model.predict(X_test)
    suit_rmse = np.sqrt(mean_squared_error(y_suit_test, y_suit_pred))
    suit_r2 = r2_score(y_suit_test, y_suit_pred)
    
    # Save models and preprocessing objects
    model_data = {
        'crop_model': crop_model,
        'suit_model': suit_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': X.columns.tolist(),
        'numerical_cols': numerical_cols
    }
    
    joblib.dump(model_data, os.path.join('models', 'crop_prediction_model.joblib'))
    
    # Feature importance analysis
    crop_feature_importance = pd.DataFrame(
        crop_model.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    suit_feature_importance = pd.DataFrame(
        suit_model.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    # Generate correlation matrix
    corr = data[numerical_cols + ['suitability']].corr()
    
    # Create evaluation plots directory
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
    
    # Plot feature importance for crop prediction
    plt.figure(figsize=(10, 6))
    sns.barplot(x=crop_feature_importance.importance, y=crop_feature_importance.index)
    plt.title('Feature Importance for Crop Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', 'crop_feature_importance.png'))
    
    # Plot feature importance for suitability prediction
    plt.figure(figsize=(10, 6))
    sns.barplot(x=suit_feature_importance.importance, y=suit_feature_importance.index)
    plt.title('Feature Importance for Suitability Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', 'suit_feature_importance.png'))
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', 'correlation_matrix.png'))
    
    # Return evaluation metrics
    results = {
        'crop_accuracy': float(crop_accuracy),
        'suitability_rmse': float(suit_rmse),
        'suitability_r2': float(suit_r2),
        'message': 'Model training completed successfully'
    }
    
    return results

if __name__ == "__main__":
    results = train_model()
    print(results)