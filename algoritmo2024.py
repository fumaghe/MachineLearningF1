import pandas as pd

# Carica i file CSV
circuits = pd.read_csv('circuits24.csv')
constructor_standings = pd.read_csv('constructor_standings24.csv')
constructors = pd.read_csv('constructors24.csv')
driver_standings = pd.read_csv('driver_standings24.csv')
drivers = pd.read_csv('drivers24.csv')
races = pd.read_csv('races24.csv')
results = pd.read_csv('results24.csv')
status = pd.read_csv('status24.csv')
final_data_sorted = pd.read_csv('final_data_sorted.csv')

# Filtra i dati per le gare del 2024
races_2024 = races[races['year'] == 2024]
results_2024 = results[results['raceId'].isin(races_2024['raceId'])]

# Aggiungi la colonna 'circuitId' a 'results_2024' tramite un join con 'races'
results_2024 = results_2024.merge(races[['raceId', 'circuitId', 'name', 'date']], on='raceId', how='left')

# Unisci i dati del 2024 con i dettagli dei piloti e delle scuderie
results_2024 = results_2024.merge(drivers, on='driverId', how='left')
results_2024 = results_2024.merge(constructors, on='constructorId', how='left')
results_2024 = results_2024.merge(circuits[['circuitId', 'circuitRef', 'name', 'location', 'country']], on='circuitId', how='left')

# Rinomina o elimina le colonne duplicate prima del merge con 'races'
races = races.rename(columns={'url': 'race_url'})
results_2024 = results_2024.merge(races[['raceId', 'year', 'round', 'race_url']], on='raceId', how='left')
results_2024 = results_2024.merge(status[['statusId', 'status']], on='statusId', how='left')

# Seleziona le colonne finali da `final_data_sorted` e riordina le colonne nel DataFrame `results_2024`
final_columns = final_data_sorted.columns

# Assicura che tutte le colonne siano presenti in results_2024
for col in final_columns:
    if col not in results_2024.columns:
        results_2024[col] = 0

# Riordina le colonne per corrispondere a final_data_sorted
results_2024 = results_2024[final_columns]

# Riempi i valori mancanti con i dati del 2023, se disponibili
for col in ['grid', 'positionOrder', 'points', 'laps']:
    if col in results_2024.columns:
        results_2024[col] = results_2024.groupby('driverId')[col].transform(lambda x: x.fillna(method='ffill'))

# Unisci i dati del 2024 con final_data_sorted
final_data_updated = pd.concat([final_data_sorted, results_2024], ignore_index=True)

# Salva il nuovo dataset
final_data_updated.to_csv('final_data_2024.csv', index=False)

# Mostra le prime righe del nuovo dataset per verifica
print(final_data_updated.head())