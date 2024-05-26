import streamlit as st
from connection import load_data

# Cargar los datos desde la base de datos SQLite
df = load_data('./database.db')


# Descripción general del script
st.title("Análisis y Visualización de Préstamos de Libros en la Biblioteca")


# Contar el número de préstamos por ubicación de la biblioteca
library_loan_counts = df[df['Library Name'] !=
                         'Unknown']['Library Name'].value_counts()

# Descripción: Este análisis cuenta el número de préstamos por cada ubicación de la biblioteca, excluyendo aquellos con nombres desconocidos.
st.subheader("Número de Préstamos por Ubicación de la Biblioteca")
st.markdown("""**Descripción:** Este análisis cuenta el número de préstamos por cada ubicación de la biblioteca, excluyendo aquellos con nombres desconocidos.""")
st.markdown("**Propósito:** Visualizar qué ubicaciones de la biblioteca tienen la mayor cantidad de préstamos, lo que puede ayudar a identificar las ubicaciones más populares o con mayor demanda.")
st.bar_chart(library_loan_counts)

# Identificar los 10 títulos de libros más demandados
top_books_list = df['Title'].value_counts().head(10).index

# Filtrar los datos para incluir solo los libros más demandados y ubicaciones conocidas de la biblioteca
top_books_data = df[df['Title'].isin(
    top_books_list) & (df['Library Name'] != 'Unknown')]

# Contar el número de préstamos de los libros más demandados por ubicación de la biblioteca
library_book_counts = top_books_data.groupby(
    ['Title', 'Library Name']).size().unstack(fill_value=0)

# Descripción: Este análisis agrupa los datos de los 10 libros más demandados por ubicación de la biblioteca y cuenta el número de préstamos.
st.subheader(
    "Número de Préstamos de los 10 Libros Más Demandados por Ubicación")
st.markdown("""
**Descripción:** Este análisis agrupa los datos de los 10 libros más demandados por ubicación de la biblioteca y cuenta el número de préstamos.
""")
st.markdown("**Propósito:** Visualizar la distribución de los préstamos de los libros más populares en diferentes ubicaciones de la biblioteca, lo que puede ayudar a entender las preferencias de los usuarios en distintas ubicaciones.")
st.write(library_book_counts)
st.bar_chart(library_book_counts)
