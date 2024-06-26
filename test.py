import pandas as pd

data = pd.read_csv('predicted_results6.csv')

def convert_to_ordinal(data):
    data['predicted_rank'] = data.groupby('raceId')['predicted_positionOrder'].rank(method='first').astype(int)
    return data

def calculate_accuracy(data):
    data = convert_to_ordinal(data)
    total_predictions = len(data)
    correct_predictions = (data['predicted_rank'] == data['positionOrder']).sum()
    within_one_position = (abs(data['predicted_rank'] - data['positionOrder']) <= 1).sum()
    accuracy = (correct_predictions / total_predictions) * 100
    accuracy_within_one = (within_one_position / total_predictions) * 100
    return correct_predictions, total_predictions, accuracy, within_one_position, accuracy_within_one

correct_predictions, total_predictions, accuracy, within_one_position, accuracy_within_one = calculate_accuracy(data)

def calculate_accuracy_per_year(data):
    results = []
    for year in range(2018, 2024):
        yearly_data = data[data['year'] == year]
        correct_predictions, total_predictions, accuracy, within_one_position, accuracy_within_one = calculate_accuracy(yearly_data)
        results.append({
            'year': year,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'within_one_position': within_one_position,
            'accuracy_within_one': accuracy_within_one
        })
    return pd.DataFrame(results)

accuracy_per_year = calculate_accuracy_per_year(data)

print("--------------------------------------------------------------------------------------------------------")
print(accuracy_per_year)
print("--------------------------------------------------------------------------------------------------------")
print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Predictions within one position: {within_one_position}")
print(f"Accuracy within one position: {accuracy_within_one:.2f}%")