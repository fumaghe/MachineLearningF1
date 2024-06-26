import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import time

# Carica i dati
data = pd.read_csv('final_data_sorted.csv')

# Funzione per preparare i dati per un anno specifico
def prepare_data_for_year(data, year):
    # Filtra i dati per i 5 anni precedenti
    train_data = data[(data['year'] >= year - 5) & (data['year'] < year)]
    
    # Filtra i dati per l'anno corrente
    test_data = data[(data['year'] == year)]
    
    return train_data, test_data

# Funzione per calcolare le prestazioni nelle ultime 10 gare
def last_10_races_performance(data, driver_id):
    last_10_races = data[data['driverId'] == driver_id].sort_values(by='raceId', ascending=False).head(10)
    avg_position = last_10_races['positionOrder'].mean()
    return avg_position if not np.isnan(avg_position) else 0

# Funzione per calcolare le prestazioni su una pista specifica
def track_performance(data, driver_id, circuit_id):
    track_races = data[(data['driverId'] == driver_id) & (data['circuitId'] == circuit_id) & (data['positionOrder'].notna())]
    avg_position = track_races['positionOrder'].mean()
    wins = (track_races['positionOrder'] == 1).sum()
    podiums = (track_races['positionOrder'] <= 3).sum()
    return avg_position if not np.isnan(avg_position) else 0, wins, podiums

# Funzione per calcolare le posizioni guadagnate o perse
def positions_gained_lost(data, driver_id):
    positions = data[data['driverId'] == driver_id].copy()
    positions['gained_lost'] = positions['grid'] - positions['positionOrder']
    avg_gained_lost = positions['gained_lost'].mean()
    return avg_gained_lost if not np.isnan(avg_gained_lost) else 0

# Funzione per creare le feature e le etichette
def create_features_and_labels(data, target_col):
    features = data[['grid']].copy()
    data['previous_position'] = data.groupby('driverId')['positionOrder'].shift(1)
    features['previous_position'] = data['previous_position']
    
    features['last_10_avg_position'] = data.apply(lambda row: last_10_races_performance(data[data['raceId'] < row['raceId']], row['driverId']), axis=1)
    features['track_avg_position'], features['track_wins'], features['track_podiums'] = zip(*data.apply(lambda row: track_performance(data[data['raceId'] < row['raceId']], row['driverId'], row['circuitId']), axis=1))
    features['avg_gained_lost'] = data.apply(lambda row: positions_gained_lost(data[data['raceId'] < row['raceId']], row['driverId']), axis=1)
    
    features.fillna(0, inplace=True)  # Riempie i valori NaN con 0
    labels = data[target_col]
    return features, labels

# Funzione per addestrare e prevedere con un modello, e calcolare l'importanza delle caratteristiche
def train_and_predict(train_data, test_data, target_col, n_estimators=100):
    X_train, y_train = create_features_and_labels(train_data, target_col)
    X_test, _ = create_features_and_labels(test_data, target_col)
    
    # Addestra il modello di RandomForest
    model = RandomForestRegressor(n_estimators=n_estimators, max_features='sqrt', max_depth=20, min_samples_split=2, min_samples_leaf=1)
    model.fit(X_train, y_train)
    
    # Prevedi sui dati di test
    y_pred = model.predict(X_test)
    
    # Calcola l'importanza delle caratteristiche
    feature_importances = model.feature_importances_
    features_list = X_train.columns.tolist()
    
    # Stampa l'importanza delle caratteristiche
    print("Importanza delle caratteristiche:")
    for feature, importance in zip(features_list, feature_importances):
        print(f"{feature}: {importance}")
    
    return y_pred, feature_importances

# Calcolo dei punti in base alla posizione predetta
def calculate_points(position):
    if position == 1:
        return 25
    elif position == 2:
        return 18
    elif position == 3:
        return 15
    elif position == 4:
        return 12
    elif position == 5:
        return 10
    elif position == 6:
        return 8
    elif position == 7:
        return 6
    elif position == 8:
        return 4
    elif position == 9:
        return 2
    elif position == 10:
        return 1
    else:
        return 0

# Predizione per ogni anno dal 2018 al 2023
predictions = []
years = range(2018, 2024)  # Include fino al 2023
feature_importances_list = []

for year in years:
    # Prepara i dati
    train_data, test_data = prepare_data_for_year(data, year)
    
    # Controlla se ci sono dati sufficienti per addestramento e test
    if train_data.empty or test_data.empty:
        print(f'Anno {year}: dati insufficienti per addestramento o test.')
        continue
    
    # Prevedi le posizioni e calcola l'importanza delle caratteristiche
    start_time = time.time()
    test_data['predicted_positionOrder'], feature_importances = train_and_predict(train_data, test_data, 'positionOrder', n_estimators=100)
    feature_importances_list.append(feature_importances)
    end_time = time.time()
    print(f"Anno {year}: Tempo di esecuzione per il training e predizione: {end_time - start_time:.2f} secondi")
    
    # Calcola i punti in base alle posizioni predette
    test_data['predicted_resultPoints'] = test_data['predicted_positionOrder'].apply(calculate_points)
    
    # Calcola i punti per i costruttori
    constructor_points = test_data.groupby('constructorId')['predicted_resultPoints'].sum().reset_index()
    constructor_points.columns = ['constructorId', 'predicted_constructorPoints']
    test_data = test_data.merge(constructor_points, on='constructorId', how='left')
    
    # Aggiungi l'anno per coerenza
    test_data['year'] = year
    
    # Aggiungi le predizioni alla lista
    predictions.append(test_data)

# Combina tutte le predizioni in un unico DataFrame
if predictions:
    all_predictions = pd.concat(predictions)
    # Verifica la struttura del DataFrame
    print("DataFrame combinato:")
    print(all_predictions.head())
    print(all_predictions.columns)
    
        # Salva le predizioni in un nuovo CSV
    try:
        output_path = os.path.join(os.getcwd(), 'predicted_results5.csv')
        all_predictions.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Le predizioni sono state salvate in '{output_path}'.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file CSV: {e}")
else:
    print('Nessuna predizione disponibile.')

# Media delle importanze delle caratteristiche
if feature_importances_list:
    avg_feature_importances = np.mean(feature_importances_list, axis=0)
    print("Media delle importanze delle caratteristiche nel periodo 2018-2023:")
    features_list = ['grid', 'previous_position', 'last_10_avg_position', 'track_avg_position', 'track_wins', 'track_podiums', 'avg_gained_lost']
    for feature, importance in zip(features_list, avg_feature_importances):
        print(f"{feature}: {importance}")