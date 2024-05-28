import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
from connection import load_data

# Función para mapear valores codificados aproximados


def map_approx(value, mapping):
    return mapping.get(int(round(value)), 'Desconocido')


# Cargar los datos desde la base de datos
df = load_data('./database.db')

# Preprocesamiento de datos
df_cleaned = df.dropna(subset=['Library Name', 'Clasification', 'Loan Date'])

# Convertir la columna Loan Date a datetime
df_cleaned['Loan Date'] = pd.to_datetime(df_cleaned['Loan Date'])

# Convertir variables categóricas a numéricas
df_cleaned['Clasification_Encoded'] = df_cleaned['Clasification'].astype(
    'category').cat.codes
df_cleaned['Library_Name_Encoded'] = df_cleaned['Library Name'].astype(
    'category').cat.codes

# Seleccionar características para el clustering
features = df_cleaned[['Clasification_Encoded', 'Library_Name_Encoded']]

# Normalizar las características
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Añadir las etiquetas de los clusters al dataframe original
df_cleaned['Cluster'] = clusters

# Obtener los centroides de los clusters
centroids = kmeans.cluster_centers_

# Desnormalizar los centroides para interpretar mejor los resultados
centroids_descaled = scaler.inverse_transform(centroids)

# Crear un DataFrame con los centroides desnormalizados
centroids_df = pd.DataFrame(centroids_descaled, columns=[
                            'Clasification_Encoded', 'Library_Name_Encoded'])

# Mapear valores codificados a valores originales
clasification_mapping = dict(
    enumerate(df_cleaned['Clasification'].astype('category').cat.categories))
library_mapping = dict(
    enumerate(df_cleaned['Library Name'].astype('category').cat.categories))

centroids_df['Clasification'] = centroids_df['Clasification_Encoded'].apply(
    map_approx, args=(clasification_mapping,))
centroids_df['Library_Name'] = centroids_df['Library_Name_Encoded'].apply(
    map_approx, args=(library_mapping,))

# Visualización de los clusters por título y fecha
fig = px.scatter(df_cleaned, y='Loan Date', x=[f'Título {i}' for i in range(len(df_cleaned))], color='Cluster',
                 title='Clustering de Préstamos a lo largo del Tiempo (2023)',
                 hover_data=['Title', 'Call Number', 'Library Name', 'Loan Time'])

# Ajustar el rango de fechas en el eje Y para incluir todo el año 2023
fig.update_layout(yaxis=dict(
    tickmode='array',
    tickvals=pd.date_range(start="2023-01-01", end="2023-12-31",
                           freq='MS').strftime("%Y-%m-%d").tolist(),
    ticktext=pd.date_range(start="2023-01-01", end="2023-12-31",
                           freq='MS').strftime("%b %Y").tolist()
))

# Ocultar las etiquetas del eje X y añadir una anotación explicativa
fig.update_layout(
    xaxis_title="",
    xaxis=dict(
        showticklabels=False
    ),
    annotations=[
        dict(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Cada punto representa un título de libro"
        )
    ]
)

# Presentación de que se procura hacer con el clustering

st.markdown("""## Explicación del Clustering

El análisis de clustering presentado utiliza el algoritmo **K-Means** para agrupar los libros prestados durante el periodo de examen en cinco clusters distintos. Los pasos seguidos son:

1. **Preprocesamiento de Datos**:
   - Se eliminan las filas con valores nulos en las columnas `Library Name`, `Clasification`, y `Loan Date`.
   - La columna `Loan Date` se convierte al formato datetime para facilitar el análisis temporal.

2. **Codificación y Normalización**:
   - Las variables categóricas `Clasification` y `Library Name` se convierten a códigos numéricos.
   - Los datos se normalizan para asegurar que todas las características tengan una escala comparable.

3. **Aplicación de K-Means**:
   - Se aplica el algoritmo K-Means para agrupar los datos en 5 clusters.
   - Se añaden las etiquetas de los clusters al dataframe original para facilitar el análisis posterior.

4. **Visualización**:
   - Se crea una visualización de dispersión utilizando Plotly, donde cada punto representa un préstamo de un libro.
   - El eje Y muestra las fechas de préstamo, abarcando todo el año 2023.
   - El color de los puntos indica el cluster al que pertenece cada libro.""")

st.plotly_chart(fig)


# Análisis de los clusters
cluster_summary = df_cleaned.groupby('Cluster')[['Clasification', 'Library Name']].agg({
    'Clasification': lambda x: x.value_counts().index[0],
    'Library Name': lambda x: x.value_counts().index[0]
}).reset_index()

st.write("Resumen de Clusters:")
st.dataframe(cluster_summary)


# Explicación del Clustering
st.markdown("""


### Interpretación de los Clusters

Cada cluster agrupa libros que comparten similitudes en sus características, principalmente el género (`Clasification`) y la biblioteca de origen (`Library Name`). A continuación, se presenta un resumen de los clusters identificados:

- **Cluster 0**: Predominan los libros de [Género] en la [Biblioteca].
- **Cluster 1**: Predominan los libros de [Género] en la [Biblioteca].
- **Cluster 2**: Predominan los libros de [Género] en la [Biblioteca].
- **Cluster 3**: Predominan los libros de [Género] en la [Biblioteca].
- **Cluster 4**: Predominan los libros de [Género] en la [Biblioteca].

Estos clusters permiten identificar patrones y tendencias en los préstamos de libros, facilitando la toma de decisiones sobre la gestión de colecciones y adquisiciones futuras en las bibliotecas. Además, ayudan a determinar en qué biblioteca es necesario tener libros de qué género en épocas de exámenes, optimizando así la distribución y disponibilidad de recursos.
""")
