import pandas as pd
import sqlite3

# Load the Excel file
excel_file = './data.xlsx'
df = pd.read_excel(excel_file, skiprows=2)

# Data Cleaning
# Remove rows where all elements are NaN
df = df.dropna(how='all')

# Parse dates and times
df['Loan Date'] = pd.to_datetime(df['Loan Date'], errors='coerce')
df['Loan Time'] = pd.to_datetime(
    df['Loan Time'], format='%H:%M:%S', errors='coerce').dt.time

# Display the cleaned dataframe to verify
print("Cleaned DataFrame:")
print(df.head())

# Check for any remaining NaN values in critical columns
print("\nNaN values in 'Loan Date':")
print(df['Loan Date'].isna().sum())

print("\nNaN values in 'Loan Time':")
print(df['Loan Time'].isna().sum())

# Connect to SQLite database (or create it)
conn = sqlite3.connect('database.db')

# Write the dataframe to the SQLite database
df.to_sql('library_loans', conn, if_exists='replace', index=False)

# Check if the connection was successful
if conn:
    print("Connection established and data load was successful")
else:
    print("Connection failed")

# Close the connection
conn.close()
