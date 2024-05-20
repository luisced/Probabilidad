import pandas as pd
import sqlite3

# Load the Excel file
excel_file = './data.xlsx'
df = pd.read_excel(excel_file, skiprows=2)

conn = sqlite3.connect('database.db')

# Write the dataframe to the SQLite database
df.to_sql('library_loans', conn, if_exists='replace', index=False)

if conn:
    print("Connection established successfully")
# Close the connection
conn.close()
