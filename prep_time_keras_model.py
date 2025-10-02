import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_and_prepare_data():
    df = pd.read_csv('basket_vectors.csv', index_col=0)
      
    exclude_cols = ['id', 'courierWaitTime', 'basketValue', 'releasedAt']
    categorical_cols = ['storeId', 'day_of_week', 'hour_of_day']
    target_col = 'prep_time'

    item_cols = [col for col in df.columns if col not in categorical_cols + [target_col] + exclude_cols]

    basket_vectors = df[item_cols].values
    store_ids = df['storeId'].values
    days_of_week = df['day_of_week'].values
    hours_of_day = df['hour_of_day'].values
    prep_times = df['prep_time'].values
    
    unique_stores = np.unique(store_ids)
    store_id_map = {store_id: idx for idx, store_id in enumerate(unique_stores)}
    mapped_store_ids = np.array([store_id_map[store_id] for store_id in store_ids])
    
    scaler = StandardScaler()
    basket_vectors_normalized = scaler.fit_transform(basket_vectors)
    
    print(f"Data shape: {len(prep_times)} samples")
    print(f"Basket vector dimensions: {len(item_cols)}")
    print(f"Number of unique stores: {len(unique_stores)}")
    print(f"Prep time range: {prep_times.min():.1f} - {prep_times.max():.1f} minutes")
    print(f"Mean prep time: {prep_times.mean():.1f} minutes")
    
    return (basket_vectors_normalized, mapped_store_ids, days_of_week, hours_of_day, 
            prep_times, len(item_cols), len(unique_stores), scaler, store_id_map, item_cols)

def create_model(num_items, num_stores, store_embed_dim=50, day_embed_dim=4, hour_embed_dim=8):
    
    basket_input = layers.Input(shape=(num_items,), name='basket_vector')
    store_input = layers.Input(shape=(), name='store_id', dtype='int32')
    day_input = layers.Input(shape=(), name='day_of_week', dtype='int32')
    hour_input = layers.Input(shape=(), name='hour_of_day', dtype='int32')
    
    store_embedding = layers.Embedding(
        input_dim=num_stores, 
        output_dim=20, 
        name='store_embedding'
    )(store_input)
    store_embedding = layers.Flatten()(store_embedding)
    
    day_embedding = layers.Embedding(
        input_dim=7, 
        output_dim=7, 
        name='day_embedding'
    )(day_input)
    day_embedding = layers.Flatten()(day_embedding)
    
    hour_embedding = layers.Embedding(
        input_dim=24, 
        output_dim=24, 
        name='hour_embedding'
    )(hour_input)
    hour_embedding = layers.Flatten()(hour_embedding)
    
    combined = layers.Concatenate()([
        basket_input, 
        store_embedding, 
        day_embedding, 
        hour_embedding
    ])

    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    output = layers.Dense(1, activation='linear', name='prep_time')(x)
    
    model = keras.Model(
        inputs=[basket_input, store_input, day_input, hour_input],
        outputs=output,
        name='prep_time_predictor'
    )
    
    return model

def train_model():
    (basket_vectors, store_ids, days_of_week, hours_of_day, prep_times, 
     num_items, num_stores, scaler, store_id_map, item_cols) = load_and_prepare_data()
    
    indices = np.arange(len(prep_times))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    X_train = {
        'basket_vector': basket_vectors[train_idx],
        'store_id': store_ids[train_idx],
        'day_of_week': days_of_week[train_idx],
        'hour_of_day': hours_of_day[train_idx]
    }
    y_train = prep_times[train_idx]
    
    X_test = {
        'basket_vector': basket_vectors[test_idx],
        'store_id': store_ids[test_idx],
        'day_of_week': days_of_week[test_idx],
        'hour_of_day': hours_of_day[test_idx]
    }
    y_test = prep_times[test_idx]
    
    model = create_model(num_items, num_stores)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            'best_prep_time_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nEvaluating model...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nFinal Test Metrics:")
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f} minutes")
    print(f"Test RMSE: {rmse:.4f} minutes")

    model.save('prep_time_model.h5')

    preprocessing_data = {
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'store_id_map': {str(k): int(v) for k, v in store_id_map.items()},
        'num_items': num_items,
        'num_stores': num_stores,
        'item_columns': item_cols
    }
    
    with open('preprocessing_data.json', 'w') as f:
        json.dump(preprocessing_data, f, indent=2)
    
    print(f"\nModel saved as 'prep_time_model.h5'")
    print(f"Preprocessing data saved as 'preprocessing_data.json'")
    
    return model, scaler, store_id_map, history

if __name__ == "__main__":
    print("Training prep time prediction model")
    
    model, scaler, store_id_map, history = train_model()
    
    print("\nTraining completed!")
    print("Use predict_prep_time.py to make predictions with the trained model.")