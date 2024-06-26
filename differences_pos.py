import pandas as pd

# Carica i dati delle predizioni
file = 'predicted_results6.csv'

# Leggi il file
data = pd.read_csv(file)

# Ordina i dati per gara e posizione predetta
data = data.sort_values(by=['raceId', 'predicted_positionOrder'])

# Funzione per generare l'HTML con le prime 10 posizioni per ogni gara
def generate_html(data):
    html = ""
    grouped = data.groupby(['raceId', 'raceName', 'year'])
    for name, group in grouped:
        html += f"<h2>{name[1]} ({name[2]})</h2>"
        html += "<table border='1'><tr><th>Pilota</th><th>Posizione predetta</th><th>Posizione reale</th></tr>"
        top_10 = group.head(10)
        # Assegna i valori di classifica predetta da 1 a 10
        top_10['predicted_final_position'] = range(1, len(top_10) + 1)
        for index, row in top_10.iterrows():
            html += f"<tr><td>{row['driverForename']} {row['driverSurname']}</td><td>{row['predicted_final_position']}</td><td>{row['positionOrder']}</td></tr>"
        html += "</table><br>"
    return html

# Genera l'HTML
html_content = generate_html(data)

# Salva l'HTML in un file
output_html = 'top_10_positions_per_race.html'
with open(output_html, 'w') as f:
    f.write(html_content)

print(f"I risultati delle prime 10 posizioni per gara sono stati salvati in '{output_html}'.")
