import pandas as pd
import numpy as np
import os

# Carica i dati delle predizioni
files = [
    'predicted_results3.csv',
    'predicted_results4.csv',
    'predicted_results5.csv'
]

# Leggi i file
data_frames = [pd.read_csv(file) for file in files]

# Unisci tutti i dati in un unico DataFrame per selezionare il 20% delle gare casualmente
all_data = pd.concat(data_frames)

# Ottieni una lista unica di gare
unique_races = all_data[['raceId', 'raceName', 'year']].drop_duplicates()

# Seleziona il 20% delle gare casualmente
sample_size = int(0.2 * len(unique_races))
sample_races = unique_races.sample(n=sample_size, random_state=42)

# Filtra i dati per le gare selezionate
sample_data_frames = [df[df['raceId'].isin(sample_races['raceId'])] for df in data_frames]

# Funzione per calcolare la differenza media tra predizione e realtà
def evaluate_predictions(df):
    if 'predicted_positionOrder' in df.columns:
        df = df.copy()  # Evita l'errore SettingWithCopyWarning
        df['position_diff'] = abs(df['positionOrder'] - df['predicted_positionOrder'])
        mean_diff = df.groupby(['raceId', 'raceName', 'year'])['position_diff'].mean().reset_index()
        return mean_diff
    else:
        return None

# Calcola la differenza media per ogni predizione
results = [evaluate_predictions(df) for df in sample_data_frames]

# Combina i risultati in un unico DataFrame per visualizzare l'accuratezza media per gara
combined_results = pd.concat(results, keys=[f'Prediction {i+3}' for i in range(len(results))], names=['Prediction', 'Index'])

# Calcola l'accuratezza media per ogni gara e ogni predizione
accuracy_per_race = combined_results.groupby(['raceId', 'raceName', 'year', 'Prediction'])['position_diff'].mean().unstack()

# Calcola l'accuratezza media finale per ogni predizione
final_accuracy_summary = combined_results.groupby('Prediction')['position_diff'].mean().reset_index()
final_accuracy_summary.rename(columns={'position_diff': 'mean_position_diff'}, inplace=True)

# Funzione per colorare le celle in base ai valori
def color_cells(val):
    min_val = accuracy_per_race.min().min()
    max_val = accuracy_per_race.max().max()
    color = ''
    if val == min_val:
        color = 'green'
    elif val == max_val:
        color = 'red'
    return f'background-color: {color}'

# Applica la colorazione alle tabelle
styled_race_accuracy = accuracy_per_race.style.applymap(color_cells)
styled_final_accuracy = final_accuracy_summary.style.applymap(color_cells)

# Salva i risultati in un file HTML
output_html_race = 'accuracy_summary_per_race.html'
styled_race_accuracy.to_html(output_html_race)
print(f"Il sommario dell'accuratezza per gara è stato salvato in '{output_html_race}'.")

output_html_final = 'final_accuracy_summary.html'
styled_final_accuracy.to_html(output_html_final)
print(f"Il sommario dell'accuratezza finale è stato salvato in '{output_html_final}'.")
