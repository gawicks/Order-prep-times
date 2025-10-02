from flask import Flask, render_template, request, jsonify
import json
import os
import numpy as np
from predict_prep_time import predict_single_order

app = Flask(__name__)

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_item_columns():
    """Load available item columns from preprocessing data"""
    try:
        with open('preprocessing_data.json', 'r') as f:
            preprocessing_data = json.load(f)
        return preprocessing_data['item_columns']
    except FileNotFoundError:
        return []

@app.route('/')
def index():
    item_columns = load_item_columns()
    return render_template('index.html', item_columns=item_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        store_id = int(data['store_id'])
        day_of_week = int(data['day_of_week'])
        hour_of_day = int(data['hour_of_day'])
        item_quantities = data['item_quantities']
        
        # Convert string quantities to float
        item_quantities = {k: float(v) for k, v in item_quantities.items() if v}
        
        predicted_time = predict_single_order(store_id, day_of_week, hour_of_day, item_quantities)
        
        # Convert numpy float to Python float for JSON serialization
        if isinstance(predicted_time, np.floating):
            predicted_time = float(predicted_time)
        elif isinstance(predicted_time, np.integer):
            predicted_time = int(predicted_time)
        else:
            predicted_time = float(predicted_time)
        
        return jsonify({
            'success': True,
            'predicted_prep_time': round(predicted_time, 2),
            'inputs': {
                'store_id': store_id,
                'day_of_week': day_of_week,
                'hour_of_day': hour_of_day,
                'item_quantities': item_quantities
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)