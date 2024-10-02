import pandas as pd
import datetime

input_file = "oxygen.txt"
output_file = f"{input_file.split('.txt')[0]}_cleaned.csv"

data = []

with open(input_file, 'r') as file:
    for line in file:
        # Remove extra spaces and split elements by tabs
        parts = line.strip().split('\t')
        if len(parts) == 3 and parts[0].startswith("oxygen:"):
            # Extract information
            start_time = pd.to_datetime(parts[1].strip(), format="%d/%m/%Y, %H:%M")
            spo2 = float(parts[2]) * 100 # In percentage

            # Append the extracted information to the list
            data.append([
                start_time.date(),
                start_time.strftime("%H:%M:%S"),  # Start time formatted as HH:MM:SS
                spo2
            ])

# Create a DataFrame with the specified columns
df = pd.DataFrame(data, columns=['Date', 'sampling_time', 'spo2_(%)'])

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)

print(f"The file {output_file} has been successfully created.")


print(df)
