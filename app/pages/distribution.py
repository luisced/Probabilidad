import numpy as np
import streamlit as st
from connection import load_data
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd


df = load_data('./database.db')


@st.cache_data
def plot_poisson_distribution(df=df):
    # Distribución de Préstamos por Día

    loan_counts = df['Loan Date'].value_counts()
    mean_loans_per_day = loan_counts.mean()

    # Generar distribución de Poisson
    poisson_dist = np.random.poisson(mean_loans_per_day, 1000)
    hist_data = [poisson_dist]
    group_labels = ['Poisson Distribution']

    st.title("Distribución de Poisson de la Demanda de Libros por Día")
    st.write("La distribución de Poisson es un modelo matemático que describe la probabilidad de un número de eventos en un intervalo de tiempo fijo, dado un valor promedio de eventos por intervalo.")
    st.write("En este caso, estamos modelando la demanda diaria de libros en la biblioteca utilizando la distribución de Poisson.")

    st.write("Número promedio de préstamos por día:", mean_loans_per_day)

    fig = ff.create_distplot(hist_data, group_labels)

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)


# Llamar a la función para mostrar la distribución
plot_poisson_distribution()


# Crear una columna de año-mes para el análisis de series temporales
df['YearMonth'] = df['Loan Date'].dt.to_period('M').astype(str)


# Agrupar por año-mes y contar el número de préstamos
monthly_loans = df.groupby('YearMonth').size().reset_index(name='Loan Count')

# Visualizar la cantidad de préstamos por mes utilizando Plotly
fig = px.line(monthly_loans, x='YearMonth', y='Loan Count',
              title='Cantidad de Préstamos por Mes')
st.plotly_chart(fig)
