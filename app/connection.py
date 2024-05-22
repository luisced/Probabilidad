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

# Handle missing values (e.g., filling with 'Unknown' or removing rows)
# For this example, we'll fill missing 'Library Name' and 'Title' with 'Unknown'
df['Library Name'] = df['Library Name'].fillna('Unknown')
df['Title'] = df['Title'].fillna('Unknown')

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
