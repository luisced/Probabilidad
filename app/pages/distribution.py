import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from connection import load_data
import plotly.figure_factory as ff


@st.cache_data
def plot_poisson_distribution():
    # Distribución de Préstamos por Día
    df = load_data('./database.db')

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

# # Distribución de Préstamos por Día
# loan_counts = df['Loan Date'].value_counts()
# mean_loans_per_day = loan_counts.mean()

# # Generar distribución de Poisson
# poisson_dist = np.random.poisson(mean_loans_per_day, 1000)

# # Visualizar la distribución de Poisson con streamlit

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.hist(poisson_dist, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
# ax.set_xlabel('Número de Préstamos por Día')
# ax.set_ylabel('Frecuencia')
# ax.set_title('Distribución de Poisson de la Demanda de Libros por Día')
# st.pyplot(fig)
