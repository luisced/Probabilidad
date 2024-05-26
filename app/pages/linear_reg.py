import pandas as pd
import streamlit as st
import plotly.express as px
from connection import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar y preprocesar los datos
df = load_data('./database.db')

# Convertir Loan Date a formato de fecha
df['Loan Date'] = pd.to_datetime(df['Loan Date'])

# Crear una columna de año-mes para el análisis de series temporales
df['YearMonth'] = df['Loan Date'].dt.to_period('M').astype(str)

# Rellenar los valores faltantes si es necesario
df = df.dropna(subset=['Loan Time Seconds',
               'Clasification', 'Is Exam Period', 'Library Name'])

# Obtener las materias únicas
materias = df['Clasification'].unique()

# Ajustar y visualizar la regresión lineal para cada materia
for materia in materias:
    df_materia = df[df['Clasification'] == materia]

    # Seleccionar las columnas necesarias para la regresión
    df_regression = df_materia[['Loan Time Seconds',
                                'Is Exam Period', 'Library Name', 'YearMonth']]

    # Convertir las variables categóricas a variables dummy
    df_regression = pd.get_dummies(df_regression, columns=[
                                   'Is Exam Period', 'Library Name', 'YearMonth'], drop_first=True)

    # Separar las características (X) y la variable objetivo (y)
    X = df_regression.drop('Loan Time Seconds', axis=1)
    y = df_regression['Loan Time Seconds']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Crear el modelo de regresión lineal
    model = LinearRegression()

    # Ajustar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # Realizar predicciones con el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Materia: {materia} - Error Cuadrático Medio (MSE): {mse}')

    # Crear un DataFrame para la visualización
    df_visualization = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Visualizar la regresión lineal
    fig = px.scatter(df_visualization, x='Predicted', y='Actual',
                     title=f'Regresión Lineal de Duración de Préstamo para {materia}')
    fig.add_traces(
        px.line(df_visualization, x='Predicted', y='Predicted').data)
    st.plotly_chart(fig)
