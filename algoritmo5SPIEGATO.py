import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import os

# Carica i dati
data = pd.read_csv('final_data_sorted.csv')

# Verifica delle colonne
# Assicuriamoci che il DataFrame contenga tutte le colonne necessarie
expected_columns = ['driverId', 'grid', 'positionOrder', 'year', 'constructorId']
missing_columns = [col for col in expected_columns if col not in data.columns]

if missing_columns:
    raise KeyError(f"Le seguenti colonne mancano nel DataFrame: {missing_columns}")

# Funzione per calcolare le caratteristiche aggiuntive
def calculate_additional_features(data):
    # Calcola la posizione precedente del pilota
    data['previous_position'] = data.groupby('driverId')['positionOrder'].shift(1)
    # Calcola la media delle ultime 10 posizioni
    data['avg_last_10_positions'] = data.groupby('driverId')['positionOrder'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    
    # Calcola le posizioni guadagnate o perse rispetto alla griglia
    data['positions_gained'] = data['grid'] - data['positionOrder']
    # Calcola la media delle posizioni guadagnate nelle ultime 10 gare
    data['avg_positions_gained'] = data.groupby('driverId')['positions_gained'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    
    # Riempi i valori NaN con 0 o altre strategie appropriate
    data.fillna(0, inplace=True)
    return data

# Applica la funzione per calcolare le caratteristiche aggiuntive
data = calculate_additional_features(data)

# Funzione per preparare i dati per un anno specifico
def prepare_data_for_year(data, year):
    # Seleziona i dati di training dai 5 anni precedenti all'anno corrente
    train_data = data[(data['year'] >= year - 5) & (data['year'] < year)]
    # Seleziona i dati di test per l'anno corrente
    test_data = data[data['year'] == year]
    
    print(f"Anno {year}: {len(train_data)} dati di training, {len(test_data)} dati di test")
    
    return train_data, test_data

# Funzione per creare le feature e le etichette
def create_features_and_labels(data, target_col):
    # Seleziona le caratteristiche da usare per il modello
    features = data[['grid', 'previous_position', 'avg_last_10_positions', 'avg_positions_gained']]
    # Seleziona l'etichetta (variabile target)
    labels = data[target_col]
    return features, labels

# Funzione per addestrare e prevedere con un modello
def train_and_predict(train_data, test_data, target_col):
    # Crea le feature e le etichette per il training
    X_train, y_train = create_features_and_labels(train_data, target_col)
    # Crea le feature per il test (le etichette non sono necessarie qui)
    X_test, _ = create_features_and_labels(test_data, target_col)
    
    # Pipeline per preprocessamento e modello
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Imputazione dei valori mancanti
        ('scaler', StandardScaler()),  # Standardizzazione delle feature
        ('model', RandomForestRegressor())  # Modello di RandomForest
    ])
    
    # Definizione della griglia di iperparametri per la grid search
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_features': ['sqrt', 'log2'],
        'model__max_depth': [10, 20, 30],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
    
    # Implementazione della grid search con cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Miglior modello trovato dalla grid search
    best_model = grid_search.best_estimator_
    # Previsione sui dati di test
    y_pred = best_model.predict(X_test)
    
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

# Predizione per ogni anno dal 2018 al 2023
predictions = []
years = range(2018, 2024)

for year in years:
    # Prepara i dati per l'anno specificato
    train_data, test_data = prepare_data_for_year(data, year)
    
    if train_data.empty or test_data.empty:
        print(f'Anno {year}: dati insufficienti per addestramento o test.')
        continue
    
    try:
        # Prevedi le posizioni finali dei piloti
        test_data['predicted_positionOrder'] = train_and_predict(train_data, test_data, 'positionOrder')
        # Calcola i punti in base alle posizioni predette
        test_data['predicted_resultPoints'] = test_data['predicted_positionOrder'].apply(calculate_points)
        
        # Calcola i punti per i costruttori
        constructor_points = test_data.groupby('constructorId')['predicted_resultPoints'].sum().reset_index()
        constructor_points.columns = ['constructorId', 'predicted_constructorPoints']
        # Aggiungi i punti predetti dei costruttori ai dati di test
        test_data = test_data.merge(constructor_points, on='constructorId', how='left')
        
        # Aggiungi l'anno per coerenza
        test_data['year'] = year
        
        # Aggiungi le predizioni alla lista
        predictions.append(test_data)
    except Exception as e:
        print(f"Errore durante la predizione per l'anno {year}: {e}")

# Combina tutte le predizioni in un unico DataFrame e salva i risultati
if predictions:
    all_predictions = pd.concat(predictions)
    print("DataFrame combinato:")
    print(all_predictions.head())
    print(all_predictions.columns)
    
    try:
        # Salva i risultati in un file CSV
        all_predictions.to_csv('predicted_results6.csv', index=False)
        print("Le predizioni sono state salvate in 'predicted_results6.csv'.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")
else:
    print('Nessuna predizione disponibile.')
