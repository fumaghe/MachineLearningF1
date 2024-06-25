import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

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
        for col in ['grid', 'constructorPoints', 'driverPoints', 'laps', 'constructorPosition', 'constructorWins', 'driverPosition', 'driverWins', 'resultPoints']:
            test_data[col] = np.nan
        # Usa valori medi per le colonne mancanti
        test_data = test_data.fillna(train_data.mean())
    else:
        # Filtra i dati per l'anno corrente
        test_data = data[data['year'] == year]
        active_drivers = test_data['driverId'].unique()
        train_data = train_data[train_data['driverId'].isin(active_drivers)]
    
    return train_data, test_data

# Funzione per creare le feature e le etichette
def create_features_and_labels(data, target_col):
    # Seleziona le feature rilevanti
    features = data[['grid', 'constructorPoints', 'driverPoints', 'laps']]
    labels = data[target_col]
    return features, labels

# Funzione per addestrare e prevedere con un modello
def train_and_predict(train_data, test_data, target_col):
    X_train, y_train = create_features_and_labels(train_data, target_col)
    X_test, _ = create_features_and_labels(test_data, target_col)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred

# Colonne da prevedere
target_columns = [
    'constructorPoints', 'constructorPosition', 'constructorWins', 
    'driverPoints', 'driverPosition', 'driverWins', 
    'grid', 'positionOrder', 'resultPoints', 'laps'
]

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
    
    # Prevedi le colonne rilevanti
    for col in target_columns:
        test_data.loc[:, f'predicted_{col}'] = train_and_predict(train_data, test_data, col)
    
    # Aggiungi l'anno per coerenza
    test_data.loc[:, 'year'] = year
    
    # Aggiungi le predizioni alla lista
    predictions.append(test_data)
    
    if year != 2024:
        # Calcola l'errore per la posizione
        y_test = test_data['positionOrder']
        y_pred = test_data['predicted_positionOrder']
        mse = mean_squared_error(y_test, y_pred)
        # Salva l'errore in una lista per riferimento futuro
        print(f'Anno {year}: Mean Squared Error = {mse}')

# Combina tutte le predizioni in un unico DataFrame
if predictions:
    all_predictions = pd.concat(predictions)
    # Salva le predizioni in un nuovo CSV
    all_predictions.to_csv('predicted_results2.csv', index=False)
else:
    print('Nessuna predizione disponibile.')
