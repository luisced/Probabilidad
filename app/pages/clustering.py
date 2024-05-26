import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
from connection import load_data

# Cargar y preprocesar los datos
df = load_data('./database.db')

# Convertir las columnas necesarias a tipo datetime y calcular el tiempo de préstamo en segundos
df['Loan Date'] = pd.to_datetime(df['Loan Date'])
df['Loan Time Seconds'] = pd.to_timedelta(df['Loan Time']).dt.total_seconds()

# Filtrar por periodos de exámenes
exam_period_df = df[df['Is Exam Period'] == 1]

# Normalizar los datos antes del clustering
scaler = StandardScaler()
exam_period_df['Loan Time Seconds Normalized'] = scaler.fit_transform(
    exam_period_df[['Loan Time Seconds']])

# Preparar los datos para el clustering
X = exam_period_df[['Loan Time Seconds Normalized']]

# Aplicar KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
exam_period_df['Cluster'] = kmeans.fit_predict(X)

# Mostrar los resultados en Streamlit
st.title('Clustering de Categorías Prestadas por Mayor Tiempo en Período de Exámenes')

# Visualización de los clusters utilizando plotly
fig = px.scatter(exam_period_df, x='Clasification', y='Loan Time Seconds', color='Cluster',
                 title='Clusters de Libros Basados en Duración de Préstamo durante Períodos de Exámenes')
fig.update_layout(xaxis={'categoryorder': 'total descending'})
st.plotly_chart(fig)

# Mostrar algunos datos de ejemplo de cada cluster
for cluster_num in exam_period_df['Cluster'].unique():
    st.write(f"Cluster {cluster_num}")
    st.write(exam_period_df[exam_period_df['Cluster'] == cluster_num].head())
