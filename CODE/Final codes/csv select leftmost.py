# import csv

# def extract_extremes(csv_file, column_index):
#     min_value = float('inf')
#     max_value = float('-inf')

#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip the header row if present

#         for row in reader:
#             if len(row) > column_index:
#                 value = float(row[column_index])
#                 min_value = min(min_value, value)
#                 max_value = max(max_value, value)

#     return min_value, max_value

# # Specify the CSV file path and the column index (0-based) you want to extract extremes from
# csv_file_path = 'D:/BTP/corner_coordinates.csv'
# column_index = 2  # Change this to the desired column index

# min_val, max_val = extract_extremes(csv_file_path, column_index)
# print(f"Minimum value: {min_val}")
# print(f"Maximum value: {max_val}")



import csv

def extract_extremes(csv_file, column_index):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header row
        data = list(reader)
    
    if column_index >= len(header):
        print("Invalid column index.")
        return
    min_value = float('inf')
    max_value = float('-inf')
    min_row = None
    max_row = None
    for row in data:
        try:
            value = float(row[column_index])
            if value < min_value:
                min_value = value
                min_row = row
            if value > max_value:
                max_value = value
                max_row = row
        except ValueError:
            pass  # Ignore rows with non-numeric values in the specified column
    return min_row, max_row
csv_file = 'D:/BTP/corner_coordinates.csv'  # Replace with the path to your CSV file
column_index = 1  # Replace with the index of the column you want to extract extremes from
min_row, max_row = extract_extremes(csv_file, column_index)
if min_row:
    print(f"Minimum value in column {column_index}: {min_row[column_index]}")
    print("Corresponding row:", min_row)
else:
    print("No valid minimum value found.")
if max_row:
    print(f"Maximum value in column {column_index}: {max_row[column_index]}")
    print("Corresponding row:", max_row)
else:
    print("No valid maximum value found.")