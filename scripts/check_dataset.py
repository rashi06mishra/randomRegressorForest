import pandas as pd
import numpy as np
import os
p = os.path.join(os.path.dirname(__file__), '..', 'vehicles_dataset_5000.csv')
df = pd.read_csv(p)
print('Total rows:', len(df))
print('Null counts:')
print(df.isnull().sum())
# parse times
df['timeIn'] = pd.to_datetime(df['timeIn'], format='%d-%m-%Y %H:%M', errors='coerce')
df['timeOut'] = pd.to_datetime(df['timeOut'], format='%d-%m-%Y %H:%M', errors='coerce')
print('\nParsed time nulls: timeIn, timeOut ->', df['timeIn'].isnull().sum(), df['timeOut'].isnull().sum())
# compute duration
df['duration_minutes'] = (df['timeOut'] - df['timeIn']).dt.total_seconds() / 60
print('Duration nulls:', df['duration_minutes'].isnull().sum())
print('Duration negative count:', (df['duration_minutes'] < 0).sum())

# Show rows with problems
if df['timeIn'].isnull().any() or df['timeOut'].isnull().any():
    print('\nRows with unparsable times:')
    print(df[df['timeIn'].isnull() | df['timeOut'].isnull()].head(10))
if (df['duration_minutes'] < 0).any():
    print('\nRows with negative duration:')
    print(df[df['duration_minutes'] < 0].head(10))
else:
    print('\nNo negative durations found')
