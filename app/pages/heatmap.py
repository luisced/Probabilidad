import numpy as np
import streamlit as st
from connection import load_data
import plotly.express as px
import pandas as pd

# Cargar los datos desde la base de datos
df = load_data('./database.db')
# Parse the 'Loan Date' to datetime and extract the week number
df['Week'] = df['Loan Date'].dt.isocalendar().week

# Pivot table to create a heatmap-friendly format
pivot_table = df.pivot_table(values='Barcode', index='Week',
                             columns='Library Name', aggfunc='count', fill_value=0)

# Function to generate and display the heatmap


@st.cache_data
def plot_heatmap(pivot_table=pivot_table):
    fig = px.imshow(pivot_table,
                    labels=dict(x="Bibliotecas", y="Semana", color="Rentas"),
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    aspect="auto",
                    color_continuous_scale='YlGnBu')
    fig.update_layout(title="Rentas semanales por biblioteca")
    return fig


# Streamlit app layout
st.title("Heatmap de Rentas Semanales por Biblioteca")
st.write("Este heatmap muestra las rentas semanales por cada biblioteca, permitiendo identificar patrones y tendencias en los datos de una manera visual.")
fig = plot_heatmap()
st.plotly_chart(fig)

# Explicación del Análisis
st.markdown("""
## Explicación del Análisis

### Heatmap de Rentas Semanales por Biblioteca
Este heatmap muestra las rentas semanales por cada biblioteca. Los colores representan la intensidad de las rentas, con tonos más oscuros indicando mayores cantidades de rentas. Esta visualización ayuda a identificar patrones y tendencias en los datos, facilitando la toma de decisiones basada en el comportamiento observado.

- **Eje X (Bibliotecas)**: Representa las diferentes bibliotecas.
- **Eje Y (Semana)**: Representa las semanas del año.
- **Color (Rentas)**: Representa la cantidad de rentas, con tonos más oscuros indicando mayores valores.
""")
