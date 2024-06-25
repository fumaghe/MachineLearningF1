import pandas as pd

# Carica i dati dai vari CSV
circuits = pd.read_csv('circuits.csv')
constructor_standings = pd.read_csv('constructor_standings.csv')
constructors = pd.read_csv('constructors.csv')
driver_standings = pd.read_csv('driver_standings.csv')
drivers = pd.read_csv('drivers.csv')
races = pd.read_csv('races.csv')
results = pd.read_csv('results.csv')
status = pd.read_csv('status.csv')

# Rinomina colonne nei DataFrame per evitare conflitti
circuits.rename(columns={'name': 'circuitName', 'location': 'circuitLocation', 'country': 'circuitCountry', 'url': 'circuitUrl'}, inplace=True)
constructors.rename(columns={'name': 'constructorName', 'nationality': 'constructorNationality', 'url': 'constructorUrl'}, inplace=True)
drivers.rename(columns={'forename': 'driverForename', 'surname': 'driverSurname', 'nationality': 'driverNationality', 'url': 'driverUrl'}, inplace=True)
races.rename(columns={'name': 'raceName'}, inplace=True)
results.rename(columns={'points': 'resultPoints', 'time': 'resultTime'}, inplace=True)
status.rename(columns={'status': 'raceStatus'}, inplace=True)
driver_standings.rename(columns={'points': 'driverPoints', 'position': 'driverPosition', 'wins': 'driverWins'}, inplace=True)
constructor_standings.rename(columns={'points': 'constructorPoints', 'position': 'constructorPosition', 'wins': 'constructorWins'}, inplace=True)

# Unisci i dati su chiavi comuni
data = results.merge(races, on='raceId')
data = data.merge(drivers, on='driverId')
data = data.merge(constructors, on='constructorId')
data = data.merge(circuits, on='circuitId')
data = data.merge(status, on='statusId', how='left')

# Aggiungi le informazioni di standings
data = data.merge(driver_standings[['raceId', 'driverId', 'driverPoints', 'driverPosition', 'driverWins']], on=['raceId', 'driverId'], how='left')
data = data.merge(constructor_standings[['raceId', 'constructorId', 'constructorPoints', 'constructorPosition', 'constructorWins']], on=['raceId', 'constructorId'], how='left')

# Controlla le colonne del DataFrame finale
print(data.columns)

# Seleziona le colonne rilevanti
final_data = data[[
    'circuitId', 'circuitName', 'circuitLocation', 'circuitCountry',
    'constructorId', 'constructorName', 'constructorNationality', 'constructorPoints', 'constructorPosition', 'constructorWins',
    'driverId', 'driverForename', 'driverSurname', 'driverNationality', 'driverPoints', 'driverPosition', 'driverWins',
    'raceId', 'year', 'round', 'raceName', 'date',
    'resultId', 'grid', 'positionOrder', 'resultPoints', 'laps', 'resultTime', 'raceStatus'
]]

# Salva il CSV finale
final_data.to_csv('final_data.csv', index=False)