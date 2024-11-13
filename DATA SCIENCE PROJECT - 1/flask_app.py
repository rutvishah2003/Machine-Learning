from flask import Flask, request, jsonify
from joblib import load

# Load the trained Random Forest model
model = load('random_forest_model.joblib')

# Initialize the Flask app
app = Flask(__name__)

# Mapping for categorical values
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
}

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract and encode input features
    tenure = data.get('tenure')
    internet_service = data.get('internet_service')
    contract = data.get('contract')
    monthly_charges = data.get('monthly_charges')
    total_charges = data.get('total_charges')

    # Map categorical features to numeric values
    internet_service_encoded = label_mapping.get(internet_service, -1)
    contract_encoded = label_mapping.get(contract, -1)

    # Validate encoded values
    if internet_service_encoded == -1 or contract_encoded == -1:
        return jsonify({'error': 'Invalid input for categorical features'}), 400

    # Prepare the input data for prediction
    input_data = [[tenure, internet_service_encoded, contract_encoded, monthly_charges, total_charges]]

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Interpret the result
    result = "This customer is likely to stay." if prediction[0] == 0 else "This customer is likely to churn."

    # Return the result as JSON
    return jsonify({'prediction': result})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
