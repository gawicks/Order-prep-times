import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def predict_prep_time(basket_vector, store_id, day_of_week, hour_of_day):
    """
    Predict preparation time for a given order
    
    Args:
        basket_vector: Array of item quantities
        store_id: Store identifier
        day_of_week: Day of week (0=Monday, 6=Sunday)
        hour_of_day: Hour of day (0-23)
    
    Returns:
        Predicted preparation time in minutes
    """
    model = keras.models.load_model('prep_time_model.h5')
    
    total_weights = model.count_params()
    
    with open('preprocessing_data.json', 'r') as f:
        preprocessing_data = json.load(f)
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(preprocessing_data['scaler_mean'])
    scaler.scale_ = np.array(preprocessing_data['scaler_scale'])
    
    store_id_map = preprocessing_data['store_id_map']
    
    basket_vector_normalized = scaler.transform([basket_vector])[0]
    
    mapped_store_id = store_id_map.get(str(store_id), 0)
    
    input_data = {
        'basket_vector': np.array([basket_vector_normalized]),
        'store_id': np.array([mapped_store_id]),
        'day_of_week': np.array([day_of_week]),
        'hour_of_day': np.array([hour_of_day])
    }
    
    prediction = model.predict(input_data, verbose=0)[0][0]
    
    return prediction

def demo_prediction():
    """
    Run predictions on sample data from basket_vectors.csv
    """
    print("\nRunning predictions on rows from basket_vectors.csv:")
    
    df = pd.read_csv('validation_set_vectors.csv', index_col=0)
    
    categorical_cols = ['storeId', 'day_of_week', 'hour_of_day', 'prep_time']
    item_cols = [col for col in df.columns if col not in categorical_cols]
    
    # Load expected item columns from preprocessing data
    with open('preprocessing_data.json', 'r') as f:
        preprocessing_data = json.load(f)
    expected_item_cols = preprocessing_data['item_columns']
    
    print(f"Processing {len(df)} orders...")
    print(f"CSV has {len(item_cols)} item columns, model expects {len(expected_item_cols)}")
    
    order_results = []
    
    for idx, row in df.iterrows():
        basket = np.zeros(len(expected_item_cols))
        
        # Fill in values from CSV where columns match
        for i, expected_col in enumerate(expected_item_cols):
            if expected_col in item_cols:
                basket[i] = row[expected_col]
                
        store_id = int(row['storeId'])
        day_of_week = int(row['day_of_week'])
        hour_of_day = int(row['hour_of_day'])
        actual_prep_time = row['prep_time']
        basket_value = row['basketValue']

        predicted_prep_time = predict_prep_time(basket, store_id, day_of_week, hour_of_day)
        error = abs(actual_prep_time - predicted_prep_time)
        
        print(f"Order {idx + 1:3d}")
        print(f"Store: {store_id}")
        print(f"Day: {day_of_week} {hour_of_day:2d}:00")
        print(f"Items: {np.sum(basket > 0)}")
        print(f"Actual prep time: {actual_prep_time:.2f} min")
        print(f"Predicted prep time: {predicted_prep_time:.2f} min")
        print(f"Error: {error:.2f} min")
        print(f"Basket value: ${basket_value:.2f}")
        print("-" * 30)

        order_results.append({
            'order_id': idx + 1,
            'store_id': store_id,
            'day_of_week': day_of_week,
            'hour_of_day': hour_of_day,
            'basket_vector': basket.tolist(),
            'actual_prep_time': actual_prep_time,
            'predicted_prep_time': predicted_prep_time,
            'error': error,
            'basket_value': basket_value
        })
    
    
    predictions = [order['predicted_prep_time'] for order in order_results]
    actual_times = [order['actual_prep_time'] for order in order_results]
    errors = [order['error'] for order in order_results]

    
    predictions = np.array(predictions)
    actual_times = np.array(actual_times)
    errors = np.array(errors)
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((actual_times - predictions) ** 2))
    mape = np.mean(np.abs((actual_times - predictions) / actual_times)) * 100
    
    print(f"\n{'='*50}")
    print(f"OVERALL PREDICTION STATISTICS:")
    print(f"{'='*50}")
    print(f"Total orders processed: {len(df)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Min error: {np.min(errors):.2f} minutes")
    print(f"Max error: {np.max(errors):.2f} minutes")
    print(f"Median error: {np.median(errors):.2f} minutes")
    
    
def predict_single_order(store_id, day_of_week, hour_of_day, item_quantities):
    """
    Predict prep time for a single order with custom item quantities
    
    Args:
        store_id: Store identifier
        day_of_week: Day of week (0=Monday, 6=Sunday)
        hour_of_day: Hour of day (0-23)
        item_quantities: Dictionary of {item_name: quantity}
    
    Returns:
        Predicted preparation time in minutes
    """
    with open('preprocessing_data.json', 'r') as f:
        preprocessing_data = json.load(f)
    
    item_cols = preprocessing_data['item_columns']
    
    basket_vector = np.zeros(len(item_cols))
    for item_name, quantity in item_quantities.items():
        if item_name in item_cols:
            idx = item_cols.index(item_name)
            basket_vector[idx] = quantity
    
    return predict_prep_time(basket_vector, store_id, day_of_week, hour_of_day)

if __name__ == "__main__":
    print("Prep Time Prediction Script")
    print("This script uses the trained model to make predictions.")
    
    if not os.path.exists('prep_time_model.h5'):
        print("Error: Model file 'prep_time_model.h5' not found. Please train the model first.")
        exit(1)
    
    if not os.path.exists('preprocessing_data.json'):
        print("Error: Preprocessing data file 'preprocessing_data.json' not found. Please train the model first.")
        exit(1)
    
    demo_prediction()
    