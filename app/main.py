import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from connection import load_data


def main():

    # Set Streamlit page title and sidebar
    st.title("Library Loan Optimization")
    # st.sidebar.header("Settings")

    # Page navigation
    st.page_link("pages/library_location.py",
                 label="Library Location", icon="🏫")
    st.page_link("pages/inventory.py",
                 label="Inventory Optimization", icon="📚")

    st.page_link("pages/clustering.py",
                 label="Clustering", icon="🔢"
                 )
    st.page_link("pages/distribution.py",
                 label="Distribution", icon="📦")

    st.page_link("pages/san_agustin.py",
                 label="San Agustín", icon="📚")

    # Load data
    df = load_data('./database.db')

    st.header("Introducción")
    st.write("El objetivo del proyecto es determinar qué biblioteca tiene la mayor demanda de libros durante los periodos de exámenes y cuáles son las materias más solicitadas.")

    # Display the data
    st.subheader("Data from SQLite Database")
    st.write(df)

    st.header("Descripción de la Base de Datos")
    st.write("""
    La base de datos contiene información sobre los préstamos de libros de la bilbioteca de la UP. Las columnas incluyen:
    - **Barcode:** Código de barras del libro.
    - **Call Number:** Número de clasificación del libro en la biblioteca.
    - **In House Loan Indicator:** Indicador de préstamo interno (por ejemplo, si el libro fue consultado dentro de la biblioteca).
    - **Library Name:** Nombre de la biblioteca donde se realizó el préstamo.
    - **Title:** Título del libro prestado.
    - **Loan Date:** Fecha en que se realizó el préstamo.
    - **Loan Time:** Hora en que se realizó el préstamo.
    """)


if __name__ == "__main__":
    main()
