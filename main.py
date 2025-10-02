import math
import pandas as pd
import json
import os
import plotly.express as px
import zipfile
 
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists('./orders.csv.zip'):
    print("Extracting orders.csv from orders.zip...")
    with zipfile.ZipFile('./orders.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Extraction completed.")

orders_dataset = pd.read_csv('./orders.csv')
posplu_dataset = pd.read_csv('./posplus.csv')
missing_plus = set()

def main():
    global orders_dataset
    print('hello world!')

    print(f'total number of rows read {len(orders_dataset.index)}')
    
    orders_dataset = orders_dataset.dropna(subset=['storeId'])
    print(f'rows after filtering out missing storeId: {len(orders_dataset.index)}')
    
    orders_dataset['basket'] = orders_dataset['items'].apply(parse_items_json)
    
    basket_vectors_df = create_basket_vectors(orders_dataset)
    print(f'Created basket vectors with shape: {basket_vectors_df.shape}')
    print('Sample of basket vectors:')
    print(basket_vectors_df.head(5))

    filtered_df = basket_vectors_df.copy()
    
    # Filter out data older than 6 months
    from datetime import datetime, timedelta
    six_months_ago = datetime.now(filtered_df['releasedAt'].iloc[0].tzinfo) - timedelta(days=180)
    filtered_df = filtered_df[filtered_df['releasedAt'] >= six_months_ago].copy()
    print(f'Rows after filtering data from last 6 months: {len(filtered_df.index)} (removed {len(basket_vectors_df.index) - len(filtered_df.index)} rows)')
    
    # Find store ID with highest number of samples
    store_counts = filtered_df['storeId'].value_counts()
    highest_sample_store = store_counts.index[0]
    highest_sample_count = store_counts.iloc[0]
    print(f'Store ID with highest number of samples: {highest_sample_store} ({highest_sample_count} samples)')
    print('Top 10 stores by sample count:')
    print(store_counts.head(10))

    filtered_df = filtered_df[filtered_df['storeId'] == 1076].copy()
    print(f'Rows after filtering for store ID 1076: {len(filtered_df.index)} (removed {len(basket_vectors_df.index) - len(filtered_df.index)} rows)')
    
    filtered_df = filtered_df[filtered_df['courierWaitTime'] >= 2].copy()
    print(f'Rows after filtering courierWaitTime >= 2 minutes: {len(filtered_df.index)} (removed {len(basket_vectors_df.index) - len(filtered_df.index)} rows)')
    
    # Create basket value bins for outlier detection
    filtered_df['basketValue_bin'] = pd.cut(filtered_df['basketValue'], bins=20, duplicates='drop')
    
    cleaned_dfs = []
    total_outliers_removed = 0
    
    for bin_range in filtered_df['basketValue_bin'].unique():
        if pd.isna(bin_range):
            continue
            
        bin_data = filtered_df[filtered_df['basketValue_bin'] == bin_range].copy()
        
        if len(bin_data) < 10:  # Skip bins with too few data points
            cleaned_dfs.append(bin_data)
            continue
        
        # Calculate IQR for prep_time within this basket value bin
        Q1 = bin_data['prep_time'].quantile(0.25)
        Q3 = bin_data['prep_time'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers within this bin
        bin_cleaned = bin_data[(bin_data['prep_time'] >= lower_bound) & (bin_data['prep_time'] <= upper_bound)].copy()
        bin_outliers = len(bin_data) - len(bin_cleaned)
        total_outliers_removed += bin_outliers
        
        cleaned_dfs.append(bin_cleaned)
    
    # Combine all cleaned bins
    filtered_df = pd.concat(cleaned_dfs, ignore_index=True)
    filtered_df = filtered_df.drop('basketValue_bin', axis=1)  # Remove the temporary bin column
    
    print(f'Rows after removing prep_time outliers per basket value: {len(filtered_df.index)} (removed {total_outliers_removed} outliers)')
    print(f'Prep time range after outlier removal: {filtered_df["prep_time"].min():.2f} - {filtered_df["prep_time"].max():.2f} minutes')

    filtered_df.to_csv('basket_vectors.csv')
    
    
    # Plot prep time against basketValue with interactive hover
    fig = px.scatter(filtered_df, 
                    x='basketValue', 
                    y='prep_time',
                    hover_data=['id', 'basketValue', 'prep_time'],
                    labels={'basketValue': 'Basket Value', 'prep_time': 'Prep Time (minutes)'},
                    title='Prep Time vs Basket Value (Courier Wait Time >= 2 min)')
    fig.show()

    stores = [810, 891, 813, 803]
    
    print(f'Creating plots for random stores: {stores}')
    
    for store_id in stores:
        store_data = filtered_df[filtered_df['storeId'] == store_id]
        if len(store_data) > 0:
            fig_store = px.scatter(store_data,
                                 x='basketValue', 
                                 y='prep_time',
                                 hover_data=['id', 'basketValue', 'prep_time'],
                                 labels={'basketValue': 'Basket Value', 'prep_time': 'Prep Time (minutes)'},
                                 title=f'Store {store_id}: Prep Time vs Basket Value ({len(store_data)} orders)')
            fig_store.show()

    print('Goodbye!')
    

def parse_items_json(items_str):
    try:
        return json.loads(items_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

def plu_to_container_id(plu):
    global posplu_dataset
    matching_rows = posplu_dataset[posplu_dataset['posPlu'] == plu]
    container_id = matching_rows['containerId'].iloc[0] if not matching_rows.empty else None
    if container_id == None:
        missing_plus.add(plu)

    return container_id

def flatten_basket_items(basket):
    flattened_items = defaultdict(int)
    
    for item in basket:
        pos_plu = item.get('posPlu')
        quantity = item.get('quantity', 1)
        parts = item.get('parts', [])

        if pos_plu and not parts:
            flattened_items[pos_plu] += quantity

        for part in parts:
            part_pos_plu = part.get('posPlu')
            part_quantity = part.get('quantity', 1)
            if part_pos_plu:
                flattened_items[part_pos_plu] += part_quantity
    
    return dict(flattened_items)

def create_basket_vectors(orders_df):
    orders_df['releasedAt'] = pd.to_datetime(orders_df['releasedAt'], format='mixed', utc=True)
    orders_df['pickedUpAt'] = pd.to_datetime(orders_df['pickedUpAt'], format='mixed', utc=True)
    
    print(orders_df['releasedAt'])
    print(orders_df['pickedUpAt'])
    print((orders_df['pickedUpAt'] - orders_df['releasedAt']))
    orders_df = orders_df.dropna(subset=['releasedAt', 'pickedUpAt']).copy()
    print(f'rows after filtering out missing pickedUpAt: {len(orders_df.index)}')

    orders_df['prep_time'] = ((orders_df['pickedUpAt'] - orders_df['releasedAt']).dt.total_seconds() / 60)
    orders_df['day_of_week'] = orders_df['releasedAt'].dt.dayofweek
    orders_df['hour_of_day'] = orders_df['releasedAt'].dt.hour
    
    flattened_baskets = []
    for basket in orders_df['basket']:
        flattened = flatten_basket_items(basket)
        flattened_baskets.append(flattened)
    
    print(f'missing plus {missing_plus}')
    
    # Get all posPlu values from posplus.csv instead of from orders
    global posplu_dataset
    posplus_pos_plus = sorted(posplu_dataset['posPlu'].dropna().unique())
    print(f'Total unique posPlu from posplus.csv: {len(posplus_pos_plus)}')
    
    basket_vectors = []
    for i, basket in enumerate(flattened_baskets):
        vector = [basket.get(pos_plu, 0) for pos_plu in posplus_pos_plus]
        row = orders_df.iloc[i]
        prep_time = row['prep_time']
        if(prep_time <= 1):
            continue
        vector.extend([
            int(row['storeId']),
            int(row['day_of_week']),
            int(row['hour_of_day']),
            int(row['courierWaitTime']),
            row['releasedAt'],
            prep_time,
            float(row['basketValue']),
            row['id']
        ])
        basket_vectors.append(vector)
    
    pos_plu_columns = [f'{pos_plu}' for pos_plu in posplus_pos_plus]
    additional_columns = ['storeId', 'day_of_week', 'hour_of_day', 'courierWaitTime', 'releasedAt', 'prep_time', 'basketValue', 'id']
    all_columns = pos_plu_columns + additional_columns
    
    df = pd.DataFrame(basket_vectors, columns=all_columns)

    return df

if __name__ == "__main__":
    main()