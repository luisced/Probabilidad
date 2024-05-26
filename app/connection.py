import streamlit as st
import pandas as pd
import sqlite3
from utils import is_exam_period


def preprocess_data(df):
    # # Mostrar datos originales
    # st.write("Datos originales:")
    # st.write(df.head())
    # st.write("Tamaño del DataFrame original:", df.shape)

    exam_periods = [
        # Mediados de Septiembre a principios de octubre
        ('2023-09-18', '2023-10-10'),
        # Finales de octubre a mediados de noviembre
        ('2023-10-25', '2023-11-15'),
        # Finales de noviembre a mediados de diciembre
        ('2023-11-25', '2023-12-15'),
        # Finales de febrero a mediados de marzo
        ('2023-02-25', '2023-03-15'),
        # Finales de marzo a mediados de abril
        ('2023-03-25', '2023-04-15'),
        # Mediados de mayo a fin de mayo
        ('2023-05-15', '2023-05-31'),
    ]

    exam_periods = [(pd.to_datetime(start), pd.to_datetime(end))
                    for start, end in exam_periods]
    # Convertir 'Loan Date' a datetime
    df['Loan Date'] = pd.to_datetime(df['Loan Date'], errors='coerce')
    # st.write("Después de convertir 'Loan Date' a datetime:")
    # st.write(df.head())
    # st.write("Tamaño después de 'Loan Date':", df.shape)

    # Convertir 'Loan Time' a timedelta y luego a segundos
    df['Loan Time'] = pd.to_timedelta(df['Loan Time'], errors='coerce')
    df['Loan Time Seconds'] = df['Loan Time'].dt.total_seconds()
    # st.write("Después de convertir 'Loan Time' a segundos:")
    # st.write(df.head())
    # st.write("Tamaño después de 'Loan Time':", df.shape)

    # Llenar valores nulos en 'Loan Date' con la fecha mediana
    median_date = df['Loan Date'].median()
    df['Loan Date'].fillna(median_date, inplace=True)

    # Llenar valores nulos en 'Loan Time Seconds' con la mediana
    median_seconds = df['Loan Time Seconds'].median()
    df['Loan Time Seconds'].fillna(median_seconds, inplace=True)

    # Convertir 'Loan Time' a string para almacenarla correctamente en la base de datos
    df['Loan Time'] = df['Loan Time'].astype(str)

    # Verificar si aún hay valores nulos
    # st.write("Cantidad de valores nulos después de llenarlos:")
    # st.write(df.isnull().sum())

    df['Loan Date'] = df['Loan Date'].dt.date
    df['Loan Date'] = pd.to_datetime(df['Loan Date'])
    # st.write("Después de convertir 'Loan Date' a fecha:")
    # st.write(df.head())
    # st.write("Tamaño final del DataFrame:", df.shape)

    df['Is Exam Period'] = df['Loan Date'].apply(
        lambda date: is_exam_period(date, exam_periods))

    return df


# Cargar el archivo Excel
excel_file = './data.xlsx'
df = pd.read_excel(excel_file, skiprows=2)

# Conectar a la base de datos SQLite (o crearla)
conn = sqlite3.connect('database.db')

# Procesar los datos
df = preprocess_data(df)

# Escribir el DataFrame en la base de datos SQLite
df.to_sql('library_loans', conn, if_exists='replace', index=False)

# Verificar si la conexión fue exitosa
if conn:
    print("Conexión establecida y carga de datos exitosa")
else:
    print("Conexión fallida")

# Cerrar la conexión
conn.close()


@st.cache_data
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM library_loans"
    df = pd.read_sql(query, conn)
    conn.close()

    # Convertir 'Loan Date' de vuelta a datetime
    df['Loan Date'] = pd.to_datetime(df['Loan Date'], errors='coerce')

    # Convertir 'Loan Time' de vuelta a timedelta
    df['Loan Time'] = pd.to_timedelta(df['Loan Time'], errors='coerce')

    return df
