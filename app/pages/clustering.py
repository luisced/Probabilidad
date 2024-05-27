import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from connection import load_data
import plotly.express as px
import streamlit as st

# Cargar los datos
df = load_data('./database.db')

# Convertir Loan Date a formato de fecha
df['Loan Date'] = pd.to_datetime(df['Loan Date'])

# Rellenar los valores faltantes si es necesario
df = df.dropna(subset=['Loan Time Seconds',
               'Clasification', 'Is Exam Period', 'Library Name'])

# Seleccionar las columnas necesarias para clustering
df_clustering = df[['Loan Time Seconds', 'Is Exam Period', 'Library Name']]

# Convertir las variables categóricas a variables dummy
df_clustering = pd.get_dummies(df_clustering, columns=[
                               'Is Exam Period', 'Library Name'], drop_first=True)
# Cargar los datos
df = load_data('./database.db')

# Convertir Loan Date a formato de fecha
df['Loan Date'] = pd.to_datetime(df['Loan Date'])

# Rellenar los valores faltantes si es necesario
df = df.dropna(subset=['Loan Time Seconds',
               'Clasification', 'Is Exam Period', 'Library Name'])

# Seleccionar las columnas necesarias para clustering
df_clustering = df[['Loan Time Seconds',
                    'Is Exam Period', 'Library Name', 'Clasification']]

# Convertir las variables categóricas a variables dummy
df_clustering = pd.get_dummies(df_clustering, columns=[
                               'Is Exam Period', 'Library Name', 'Clasification'], drop_first=True)

# Escalar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)

# Definir el número de clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Ajustar el modelo
kmeans.fit(df_scaled)

# Añadir las etiquetas de los clusters al DataFrame original
df['Cluster'] = kmeans.labels_

# Visualizar los clusters
fig = px.scatter(df, x='Loan Time Seconds', y='Cluster', color='Cluster',
                 title='Clustering de Duración de Préstamo')
st.plotly_chart(fig)

# Visualización de los clusters por biblioteca y duración de préstamo
fig = px.scatter(df, x='Library Name', y='Loan Time Seconds', color='Cluster',
                 title='Clustering por Biblioteca y Duración de Préstamo')
st.plotly_chart(fig)

# Visualización de los clusters por periodo de examen y duración de préstamo
fig = px.scatter(df, x='Is Exam Period', y='Loan Time Seconds', color='Cluster',
                 title='Clustering por Periodo de Examen y Duración de Préstamo')
st.plotly_chart(fig)

# Visualización de los clusters por clasificación y duración de préstamo
fig = px.scatter(df, x='Clasification', y='Loan Time Seconds', color='Cluster',
                 title='Clustering por Clasificación y Duración de Préstamo')
st.plotly_chart(fig)
