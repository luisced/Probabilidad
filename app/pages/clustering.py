import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
from connection import load_data


# Cargar y preprocesar los datos
df = load_data('./database.db')
exam_df = df[df['Is Exam Period'] == True]

# Seleccionar características relevantes
features = exam_df[['Title', 'Loan Time Seconds']]
features = features.groupby('Title').mean().reset_index()

# Normalizar las características
scaler = StandardScaler()
features[['Loan Time Seconds']] = scaler.fit_transform(
    features[['Loan Time Seconds']])

# Aplicar K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
features['Cluster'] = kmeans.fit_predict(features[['Loan Time Seconds']])

# Visualizar los clusters
fig = px.scatter(features, x='Title', y='Loan Time Seconds', color='Cluster',
                 title='Clusters de Libros Basados en Duración de Préstamo durante Períodos de Exámenes')
fig.update_layout(xaxis={'categoryorder': 'total descending'})
st.plotly_chart(fig)

# Mostrar algunos datos de ejemplo de cada cluster
for cluster_num in range(4):
    st.write(f"Cluster {cluster_num}")
    st.write(features[features['Cluster'] == cluster_num].head())
