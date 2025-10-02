import math
import pandas as pd
import json
import os
 
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))
orders_dataset = pd.read_csv('./validation_set.csv')
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

    basket_vectors_df.to_csv('validation_set_vectors.csv')
    pos_plu_columns = [col for col in basket_vectors_df.columns if col not in ['storeId', 'day_of_week', 'hour_of_day']]
    print(",".join(pos_plu_columns))
    

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