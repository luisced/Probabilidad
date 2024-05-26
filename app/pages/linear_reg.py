import statsmodels.api as sm
import plotly.express as px
import pandas as pd
import streamlit as st
from connection import load_data


def plot_regression():

    df = load_data('./database.db')

    # Convertir 'Loan Date' a formato numérico
    df['Loan Date Numeric'] = df['Loan Date'].map(pd.Timestamp.toordinal)

    # Modelo de Regresión Lineal
    X = df[['Loan Date Numeric']]
    y = df['Loan Time Seconds']
    X = sm.add_constant(X)  # Añadir constante para la ordenada al origen

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    # Resumen del Modelo
    st.write(model.summary())

    # Graficar Resultados
    fig = px.scatter(df, x='Loan Date Numeric',
                     y='Loan Time Seconds', trendline='ols')
    st.plotly_chart(fig)


# Llamar a la función para mostrar la regresión lineal
plot_regression()
