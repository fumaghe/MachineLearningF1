import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Carica i dati
data = pd.read_csv('final_data_sorted.csv')

# Elenco dei piloti per il 2024
drivers_2024 = [
    'Max VERSTAPPEN', 'Lando NORRIS', 'Charles LECLERC', 'Carlos SAINZ', 'Sergio PEREZ',
    'Oscar PIASTRI', 'George RUSSELL', 'Lewis HAMILTON', 'Fernando ALONSO', 'Yuki TSUNODA',
    'Lance STROLL', 'Daniel RICCIARDO', 'Nico HULKENBERG', 'Pierre GASLY', 'Esteban OCON',
    'Alexander ALBON', 'Kevin MAGNUSSEN', 'Guanyu Zhou', 'Valtteri Bottas', 'Logan Sargeant'
]

# Mappa dei nomi dei piloti ai loro ID
driver_names = data[['driverId', 'driverForename', 'driverSurname']].drop_duplicates()
driver_names['fullName'] = driver_names['driverForename'] + ' ' + driver_names['driverSurname']

# Filtra solo i driverId dei piloti che gareggeranno nel 2024
driver_ids_2024 = driver_names[driver_names['fullName'].isin(drivers_2024)]['driverId'].unique()

# Funzione per preparare i dati per un anno specifico
def prepare_data_for_year(data, year, driver_ids_2024=None):
    # Filtra i dati per i 5 anni precedenti
    train_data = data[(data['year'] >= year - 5) & (data['year'] < year)]
    
    # Per l'anno 2024, crea un DataFrame di test vuoto con i driver_ids_2024
    if year == 2024 and driver_ids_2024 is not None:
        test_data = pd.DataFrame({'driverId': driver_ids_2024})
        # Aggiungi colonne necessarie con valori predefiniti (o mediati dai dati passati)
        for col in ['grid', 'previous_position', 'previous_points', 'laps']:
            test_data[col] = np.nan
        # Usa valori medi per le colonne mancanti
        test_data = test_data.fillna(train_data.mean())
    else:
        # Filtra i dati per l'anno corrente
        test_data = data[(data['year'] == year) & (data['driverId'].isin(driver_ids_2024))]
    
    return train_data, test_data

# Funzione per creare le feature e le etichette
def create_features_and_labels(data, target_col):
    features = data[['grid']]
    features['previous_position'] = data.groupby('driverId')['positionOrder'].shift(1)
    features['previous_points'] = data.groupby('driverId')['resultPoints'].shift(1)
    features.fillna(0, inplace=True)  # Riempie i valori NaN con 0
    labels = data[target_col]
    return features, labels

# Funzione per addestrare e prevedere con un modello
def train_and_predict(train_data, test_data, target_col):
    X_train, y_train = create_features_and_labels(train_data, target_col)
    X_test, _ = create_features_and_labels(test_data, target_col)
    
    # Addestra il modello di RandomForest
    model = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=20, min_samples_split=2, min_samples_leaf=1)
    model.fit(X_train, y_train)
    
    # Prevedi sui dati di test
    y_pred = model.predict(X_test)
    
    return y_pred

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

# Predizione per ogni anno dal 2018 al 2024
predictions = []
years = range(2018, 2025)  # Include il 2024

for year in years:
    # Prepara i dati
    train_data, test_data = prepare_data_for_year(data, year, driver_ids_2024)
    
    # Controlla se ci sono dati sufficienti per addestramento e test
    if train_data.empty or (year != 2024 and test_data.empty):
        print(f'Anno {year}: dati insufficienti per addestramento o test.')
        continue
    
    # Prevedi le posizioni
    test_data['predicted_positionOrder'] = train_and_predict(train_data, test_data, 'positionOrder')
    
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
        all_predictions.to_csv('predicted_results3.csv', index=False)
        print("Le predizioni sono state salvate in 'predicted_results3.csv'.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")
else:
    print('Nessuna predizione disponibile.')
