import pandas as pd


 # Parse the Date and start_time columns as datetime

df_sleep = pd.read_csv("sleep_cleaned.csv") #, parse_dates=['Date', 'start_time', 'end_time'])
df_sleep['start_time'] = pd.to_datetime(df_sleep['start_time'], format="%H:%M:%S").dt.strftime("%H:%M:%S")
df_sleep['end_time'] = pd.to_datetime(df_sleep['end_time'], format="%H:%M:%S").dt.strftime("%H:%M:%S")

df_oxygen = pd.read_csv("oxygen_cleaned.csv") #, parse_dates=['Date', 'start_time'])
df_oxygen['sampling_time'] = pd.to_datetime(df_oxygen['sampling_time'], format="%H:%M:%S").dt.strftime("%H:%M:%S")

print(f"{df_sleep.shape=}, {df_oxygen.shape=}")


print("In sleep :", set(df_sleep['Date'].unique()))

print("In oxygen:", set(df_oxygen['Date'].unique()))

print("Difference:", set(df_oxygen['Date'].unique()).symmetric_difference(set(df_sleep['Date'].unique())))


# Full outer join on the 'Date' column
df_join = pd.merge(df_sleep, df_oxygen, on='Date', how='left', suffixes=('_sleep', '_oxygen'))

print(df_join.shape)

print(df_join[df_join['spo2_(%)'].isna()].shape)

# Replace NaN values in 'spo2_(%)' with 95
df_join.loc[df_join['spo2_(%)'].isna(), 'spo2_(%)'] = 95

# Replace NaN values in 'sampling_time' with the corresponding 'start_time' values
df_join.loc[df_join['spo2_(%)'] == 95, 'sampling_time'] = df_join['start_time']

print(df_join[df_join['spo2_(%)'].isna()].shape)

df_join['oxygen_within_sleep_interval'] = (
    (df_join['sampling_time'] >= df_join['start_time']) & 
    (df_join['sampling_time'] <= df_join['end_time'])
)

df_join = df_join[df_join['oxygen_within_sleep_interval'] == True]

print(df_join.shape)

df_join_filtered = df_join.groupby(by=['stage_type','Date', 'start_time', 'end_time', 'duration_(s)']).agg(
    average_spo2=('spo2_(%)', 'mean'),
    spo2_samples=('spo2_(%)', 'count') 
).reset_index()

# Save the DataFrame to a CSV file
df_join_filtered.to_csv("data.csv", index=False)

df_join_filtered
