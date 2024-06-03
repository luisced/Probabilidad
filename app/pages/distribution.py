import numpy as np
import streamlit as st
from connection import load_data
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Cargar los datos desde la base de datos
df = load_data('./database.db')

# Función para generar y mostrar la distribución de Poisson


@st.cache_data
def plot_poisson_distribution(df=df):
    # Distribución de Préstamos por Día
    loan_counts = df['Loan Date'].value_counts()
    mean_loans_per_day = loan_counts.mean()

    # Generar distribución de Poisson
    poisson_dist = np.random.poisson(mean_loans_per_day, 1000)
    hist_data = [poisson_dist]
    group_labels = ['Distribución de Poisson']

    st.title("Distribución de Poisson de la Demanda de Libros por Día")
    st.write("La distribución de Poisson es un modelo matemático que describe la probabilidad de un número de eventos en un intervalo de tiempo fijo, dado un valor promedio de eventos por intervalo.")
    st.write("En este caso, estamos modelando la demanda diaria de libros en la biblioteca utilizando la distribución de Poisson.")

    st.write("Número promedio de préstamos por día:", mean_loans_per_day)

    fig = ff.create_distplot(hist_data, group_labels)
    fig.update_layout(
        xaxis_title="Número de Préstamos",
        yaxis_title="Densidad",
        title="Distribución de Poisson de la Demanda de Libros por Día"
    )

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)


def intervalos_confianza(df=df):
    # Calcular el intervalo de confianza del 95% para la cantidad de préstamos diarios
    loan_counts = df['Loan Date'].value_counts()
    mean_loans_per_day = loan_counts.mean()
    std_loans_per_day = loan_counts.std()
    n_loans = len(loan_counts)

    # Calcular el intervalo de confianza
    z = 1.96  # Z-score para el 95% de confianza
    margin_of_error = z * (std_loans_per_day / np.sqrt(n_loans))

    lower_bound = mean_loans_per_day - margin_of_error
    upper_bound = mean_loans_per_day + margin_of_error

    st.subheader(
        "Estimación Puntual del 95% para la Cantidad de Préstamos Diarios")
    st.write("El intervalo de confianza del 95% es un rango de valores que es probable que contenga el valor real de la población con un 95% de confianza. Siempre es recomendado utilizar el 95% por ser el más comúnmente utilizado.")
    st.write("En este caso, estamos calculando el intervalo de confianza para la cantidad de préstamos diarios en la biblioteca. Asi podemos obtener la estimación puntual según os prestamos diarios obtenidos")

    st.write("El margen de error es fundamemtal para la estimación puntual, ya que nos indica cuánto puede variar la estimación puntual del valor real de la población. En este caso, el margen de error es de +/-",
             margin_of_error, "préstamos diarios. Esto significa que el valor real de la cantidad de préstamos diarios en la biblioteca es probable que esté dentro de +/-", margin_of_error, "de la estimación puntual.")
    st.write("Margen de error:", margin_of_error)

    st.write(f"El intervalo de confianza del 95% para la cantidad de préstamos diarios es [{
             lower_bound:.2f}, {upper_bound:.2f}]")

    # Grafica de intervalo de confianza
    fig = ff.create_distplot([loan_counts], ['Loan Counts'], show_hist=False)
    fig.add_shape(type='line', x0=lower_bound, x1=lower_bound,
                  y0=0, y1=0.1, line=dict(color='red', width=2))
    fig.add_shape(type='line', x0=upper_bound, x1=upper_bound,
                  y0=0, y1=0.1, line=dict(color='red', width=2))

    st.plotly_chart(fig)


# Llamar a la función para mostrar la distribución
plot_poisson_distribution()

# Crear una columna de año-mes para el análisis de series temporales
df['YearMonth'] = df['Loan Date'].dt.to_period('M').astype(str)

# Agrupar por año-mes y contar el número de préstamos
monthly_loans = df.groupby('YearMonth').size().reset_index(name='Loan Count')

# Visualizar la cantidad de préstamos por mes utilizando Plotly
fig = px.line(monthly_loans, x='YearMonth', y='Loan Count',
              title='Cantidad de Préstamos por Mes')
fig.update_layout(
    xaxis_title="Mes y Año",
    yaxis_title="Cantidad de Préstamos",
    title="Cantidad de Préstamos por Mes"
)
st.plotly_chart(fig)

# Assuming you have calculated λ (lambda)
λ = 58  # Example value, replace this with your actual λ

# Generate the Poisson distribution data
x = np.arange(30, 110)  # Adjust the range as needed
y = poisson.pmf(x, λ)

# Define the value after which bars should be a different color
threshold = 63

color_disponible = (6/255, 78/255, 188/255)
color_no_disponible = (183/255, 0, 0)

# Plotting
plt.figure(figsize=(10, 5))

# Plot bars before the threshold
plt.bar(x[x < threshold], y[x < threshold],
        color=color_disponible, label=f'P(x < {threshold})')

# Plot bars after the threshold
plt.bar(x[x >= threshold], y[x >= threshold],
        color=color_no_disponible, label=f'P(x >= {threshold})')

# Add labels and title

plt.xlabel('Número de Préstamos')
plt.ylabel('Probabilidad')
plt.title('Distribución de Poisson para la Demanda de Libros')
plt.legend()
st.pyplot()


# Explicación del Análisis
st.markdown("""
## Explicación del Análisis

### Distribución de Poisson de la Demanda de Libros por Día
La distribución de Poisson es un modelo matemático utilizado para describir la probabilidad de un número de eventos ocurridos en un intervalo de tiempo fijo, dado un valor promedio de eventos por intervalo. En este caso, utilizamos la distribución de Poisson para modelar la demanda diaria de libros en la biblioteca. Este análisis nos permite entender cómo varía la demanda diaria y evaluar si los datos siguen un patrón de Poisson.

- **Eje X (Número de Préstamos)**: Representa el número de préstamos de libros en un día específico.
- **Eje Y (Densidad)**: Representa la densidad de la distribución de Poisson, indicando la probabilidad de observar un número específico de préstamos por día.

### Cantidad de Préstamos por Mes
Para el análisis de series temporales, hemos creado una columna `YearMonth` que agrupa los préstamos por año y mes. Luego, contamos el número de préstamos en cada mes y visualizamos esta información utilizando una gráfica de líneas. Esta visualización nos ayuda a identificar patrones y tendencias en la cantidad de préstamos mensuales, lo que puede ser útil para planificar la gestión de recursos en la biblioteca.

- **Eje X (Mes y Año)**: Representa los meses del año, indicando el periodo de tiempo.
- **Eje Y (Cantidad de Préstamos)**: Representa el número total de préstamos de libros en cada mes.
""")


# Llamar a la función para mostrar el intervalo de confianza
intervalos_confianza()
