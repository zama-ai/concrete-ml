import pandas as pd
from datetime import timedelta

input_file = "sleep.txt"
output_file = f"{input_file.split('.txt')[0]}_cleaned.csv"

data = []

with open(input_file, 'r', encoding="utf-8") as file:
    for line in file:
        # Remove extra spaces and split elements by tabs
        parts = line.strip().split('\t')
        if len(parts) == 4 and parts[0].startswith("sleep:"):
            # Extract information
            start_time = pd.to_datetime(parts[1], format="%d/%m/%Y, %H:%M")
            duration = float(parts[2])  # Duration in seconds
            cycle_type = int(parts[3])  # Cycle type (int)

            # Calculate the end time
            end_time = start_time + timedelta(seconds=duration)

            # Append the extracted information to the list
            data.append([
                cycle_type, 
                start_time.date(), 
                start_time.strftime("%H:%M:%S"),  # Start time formatted as HH:MM:SS
                end_time.strftime("%H:%M:%S"),    # End time formatted as HH:MM:SS
                duration
            ])

# Create a DataFrame with the specified columns
df = pd.DataFrame(data, columns=['cycle_type', 'Date', 'start_time', 'end_time', 'duration_(s)'])

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)

print(f"The file {output_file} has been successfully created.")


print(df)