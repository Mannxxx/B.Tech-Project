import csv
import matplotlib.pyplot as plt

# Function to extract (x, y) coordinates based on a threshold
def extract_coordinates(input_file, threshold):
    x_coords = []
    y_coords = []

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header row

        for row in reader:
            x = float(row[1])  # Assuming column 1 contains x-values
            y = float(row[2])  # Assuming column 2 contains y-values

            if x > threshold:
                x_coords.append(x)
                y_coords.append(y)

    return x_coords, y_coords

# Input CSV file path
input_csv_file = 'corner_coordinates.csv'  # Replace with the path to your CSV file

csv_file_path = 'D:/BTP/CODE/corner_coordinates.csv'  # Replace with the path to your CSV file

# Prompt the user to enter a threshold value
threshold = float(input("Enter the threshold value for x: "))

# Extract (x, y) coordinates based on the threshold
x_coords, y_coords = extract_coordinates(input_csv_file, threshold)

# Plot a histogram of the x-values
plt.figure(figsize=(10, 5))
plt.hist(x_coords, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('X-values')
plt.ylabel('Frequency')
plt.title('Histogram of X-values')
plt.grid(True)

plt.show()

# Create a scatter plot of (x, y) coordinates
plt.figure(figsize=(10, 5))
plt.scatter(x_coords, y_coords, marker='o', color='b', s=30)  # s is the marker size
plt.xlabel('X-values')
plt.ylabel('Y-values')
plt.title('Scatter Plot of (X, Y) Coordinates')
plt.grid(True)


plt.show()
