from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
model = joblib.load(r'C:\Users\LENOVO\Documents\Project\Banglore House Price Prediction\bangalore_house_price_prediction_rfr_model1.pkl')
scaler = joblib.load(r'C:\Users\LENOVO\Documents\Project\Banglore House Price Prediction\scaler.pkl')

# Feature columns should match the order used when training the model
feature_columns = ['bath', 'balcony', 'total_sqft_int', 'bhk', 'price_per_sqft', 
                    'availability_Ready To Move', 'area_type_Building_Amenties', 'area_type_Carpet_Area', 
                    'area_type_Land_Area', 'location_X', 'location_Y']

def predict_house_price(features):
    x = np.zeros(len(feature_columns))
    for i, feature in enumerate(features):
        x[i] = feature
    
    x_scaled = scaler.transform([x])[0]
    return model.predict([x_scaled])[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data.get(col, 0) for col in feature_columns]
    prediction = predict_house_price(features)
    return jsonify({'price': prediction})

if __name__ == '__main__':
    app.run(debug=True)
