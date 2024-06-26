import pandas as pd

# Carica i dati delle predizioni
file = 'predicted_results6.csv'

# Leggi il file
data = pd.read_csv(file)

# Verifica i nomi delle colonne
print(data.columns)

# Ordina i dati per gara e posizione predetta
data = data.sort_values(by=['raceId', 'predicted_positionOrder'])

# Calcola i punti predetti per ogni pilota e la classifica finale per piloti e costruttori
def calculate_standings(data):
    data['predicted_final_position'] = data.groupby(['raceId', 'year'])['predicted_positionOrder'].rank(method='first')
    data['predicted_points'] = data['predicted_final_position'].apply(calculate_points)
    
    # Calcola la classifica piloti predetta
    driver_standings = data.groupby(['year', 'driverId', 'driverForename', 'driverSurname', 'constructorId'])['predicted_points'].sum().reset_index()
    driver_standings = driver_standings.sort_values(by=['year', 'predicted_points'], ascending=[True, False])
    
    # Calcola la classifica costruttori predetta
    constructor_standings = data.groupby(['year', 'constructorId'])['predicted_points'].sum().reset_index()
    constructor_standings = constructor_standings.sort_values(by=['year', 'predicted_points'], ascending=[True, False])
    
    return driver_standings, constructor_standings

# Funzione per calcolare i punti in base alla posizione predetta
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

# Funzione per generare l'HTML con le differenze tra posizioni predette e reali
def generate_position_diff_html(data):
    html = ""
    grouped = data.groupby(['raceId', 'raceName', 'year'])
    for name, group in grouped:
        html += f"<h2>{name[1]} ({name[2]})</h2>"
        html += "<table border='1'><tr><th>Pilota</th><th>Posizione predetta</th><th>Posizione reale</th><th>Race Status</th><th>Diff</th></tr>"
        top_10 = group.head(10)
        top_10['diff'] = top_10['positionOrder'] - top_10['predicted_final_position']
        for index, row in top_10.iterrows():
            html += f"<tr><td>{row['driverForename']} {row['driverSurname']}</td><td>{row['predicted_final_position']}</td><td>{row['positionOrder']}</td><td>{row['raceStatus']}</td><td>{row['diff']}</td></tr>"
        html += "</table>"
        html += f"<p>Posizioni azzeccate: {sum(group['positionOrder'] == group['predicted_final_position'])}/10</p>"
        finished_group = group[group['raceStatus'] == 'Finished']
        avg_diff = (finished_group['positionOrder'] - finished_group['predicted_final_position']).mean()
        html += f"<p>Media differenza (Finished): {avg_diff:.2f}</p>"
        html += "<br>"
    return html

# Funzione per generare il report HTML completo
def generate_report_html(data):
    driver_standings, constructor_standings = calculate_standings(data)
    
    html = "<html><head><title>Analisi delle Predizioni</title></head><body>"
    
    for year in driver_standings['year'].unique():
        year_driver_standings = driver_standings[driver_standings['year'] == year]
        year_constructor_standings = constructor_standings[constructor_standings['year'] == year]
        
        html += f"<h1>Classifica finale per l'anno {year} (Predetta)</h1>"
        html += "<h2>Classifica Piloti</h2>"
        html += "<table border='1'><tr><th>Posizione</th><th>Pilota</th><th>Scuderia</th><th>Punti Predetti</th></tr>"
        for position, (index, row) in enumerate(year_driver_standings.iterrows(), start=1):
            html += f"<tr><td>{position}</td><td>{row['driverForename']} {row['driverSurname']}</td><td>{row['constructorId']}</td><td>{row['predicted_points']}</td></tr>"
        html += "</table><br>"
        
        html += "<h2>Classifica Costruttori</h2>"
        html += "<table border='1'><tr><th>Posizione</th><th>Costruttore</th><th>Punti Predetti</th></tr>"
        for position, (index, row) in enumerate(year_constructor_standings.iterrows(), start=1):
            html += f"<tr><td>{position}</td><td>{row['constructorId']}</td><td>{row['predicted_points']}</td></tr>"
        html += "</table><br>"

        html += generate_position_diff_html(data[data['year'] == year])
    
    html += "</body></html>"
    
    return html

# Genera il report HTML
report_html = generate_report_html(data)

# Salva il report in un file HTML
output_html = 'prediction_analysis_report.html'
with open(output_html, 'w') as f:
    f.write(report_html)

print(f"Il report delle analisi delle predizioni è stato salvato in '{output_html}'.")
