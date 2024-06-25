import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

# Carica il file CSV con i risultati delle predizioni
file_path = 'predicted_results2.csv'
data = pd.read_csv(file_path)

# Funzione per filtrare i dati per anno e gara
def filter_data_by_year_and_race(data, year, race_name):
    return data[(data['year'] == year) & (data['raceName'] == race_name)].copy()

# Funzione per calcolare la differenza e colorare la tabella
def color_diff(val):
    color = 'red' if val > 0 else 'blue'
    return f'color: {color}'

# Funzione per visualizzare i dati in una tabella
def visualize_predictions_table(year, race_name):
    filtered_data = filter_data_by_year_and_race(data, year, race_name)
    
    if filtered_data.empty:
        print(f"Nessun dato disponibile per l'anno {year} e la gara {race_name}")
        return
    
    # Calcola le differenze per le metriche richieste
    for col in [
        'constructorPoints', 'constructorPosition', 'constructorWins', 
        'driverPoints', 'driverPosition', 'driverWins', 
        'grid', 'positionOrder', 'resultPoints', 'laps']:
        filtered_data.loc[:, f'diff_{col}'] = filtered_data[f'predicted_{col}'] - filtered_data[col]
    
    # Seleziona le colonne da visualizzare per i piloti
    columns_to_show_pilots = [
        'driverSurname', 'constructorName', 'grid', 'predicted_grid', 'diff_grid', 
        'positionOrder', 'predicted_positionOrder', 'diff_positionOrder', 
        'resultPoints', 'predicted_resultPoints', 'diff_resultPoints', 
        'laps', 'predicted_laps', 'diff_laps', 
        'driverPoints', 'predicted_driverPoints', 'diff_driverPoints', 
        'driverPosition', 'predicted_driverPosition', 'diff_driverPosition', 
        'driverWins', 'predicted_driverWins', 'diff_driverWins'
    ]
    
    # Seleziona le colonne da visualizzare per i costruttori
    columns_to_show_constructors = [
        'constructorName', 'constructorPoints', 'predicted_constructorPoints', 'diff_constructorPoints', 
        'constructorPosition', 'predicted_constructorPosition', 'diff_constructorPosition', 
        'constructorWins', 'predicted_constructorWins', 'diff_constructorWins'
    ]
    
    display_data_pilots = filtered_data[columns_to_show_pilots]
    display_data_constructors = filtered_data[columns_to_show_constructors].drop_duplicates()
    
    # Applica la colorazione alle differenze
    styled_data_pilots = display_data_pilots.style.applymap(color_diff, subset=[col for col in display_data_pilots.columns if col.startswith('diff_')])
    styled_data_constructors = display_data_constructors.style.applymap(color_diff, subset=[col for col in display_data_constructors.columns if col.startswith('diff_')])
    
    # Crea l'output HTML
    html_pilots = styled_data_pilots._repr_html_()
    html_constructors = styled_data_constructors._repr_html_()
    
    # Salva l'output HTML in un file
    with open(f'predictions_comparison_{year}_{race_name}.html', 'w') as f:
        f.write("<h2>Predictions vs Real for Pilots</h2>")
        f.write(html_pilots)
        f.write("<h2>Predictions vs Real for Constructors</h2>")
        f.write(html_constructors)
    
    print(f"Output salvato in predictions_comparison_{year}_{race_name}.html")

# Esempio di utilizzo della funzione
visualize_predictions_table(2023, 'Australian Grand Prix')
