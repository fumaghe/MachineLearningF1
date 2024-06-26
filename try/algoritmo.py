import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Carica i dati
data = pd.read_csv('final_data_sorted.csv')

# Funzione per preparare i dati per un anno specifico
def prepare_data_for_year(data, year):
    # Filtra i dati per i 5 anni precedenti
    train_data = data[(data['year'] >= year - 5) & (data['year'] < year)]
    
    # Filtra i dati per l'anno corrente
    test_data = data[data['year'] == year]
    
    # Rimuovi i piloti che non sono presenti nell'anno corrente
    active_drivers = test_data['driverId'].unique()
    train_data = train_data[train_data['driverId'].isin(active_drivers)]
    
    return train_data, test_data

# Funzione per creare le feature e le etichette
def create_features_and_labels(data):
    # Seleziona le feature rilevanti
    features = data[['grid', 'constructorPoints', 'driverPoints', 'laps']]
    labels = data['positionOrder']
    return features, labels

# Predizione per ogni anno dal 2018 al 2023
predictions = []
years = range(2018, 2024)

for year in years:
    # Prepara i dati
    train_data, test_data = prepare_data_for_year(data, year)
    
    # Crea le feature e le etichette
    X_train, y_train = create_features_and_labels(train_data)
    X_test, y_test = create_features_and_labels(test_data)
    
    # Addestra il modello
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Effettua le predizioni
    y_pred = model.predict(X_test)
    
    # Salva le predizioni
    test_data['predicted_position'] = y_pred
    predictions.append(test_data)
    
    # Calcola l'errore
    mse = mean_squared_error(y_test, y_pred)
    print(f'Anno {year}: Mean Squared Error = {mse}')

# Combina tutte le predizioni in un unico DataFrame
all_predictions = pd.concat(predictions)

# Salva le predizioni in un nuovo CSV
all_predictions.to_csv('predicted_results.csv', index=False)
