import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from connection import load_data
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go

st.title("Optimización de Inventario de Libros")

# Descripción de la funcionalidad
st.markdown("""
## Descripción 

**Objetivo:** Optimizar el inventario de los libros más demandados en la biblioteca.

**Metodología:**
- Filtrado y agrupación de datos por título y mes.
- Modelado de series temporales utilizando **Exponential Smoothing**.
- Cálculo del **stock de seguridad** y **punto de reorden**.

**Resultados:** Visualización de los puntos de reorden y cantidades de pedido para los 10 libros más demandados.
""")

# Cargar los datos desde la base de datos
df = load_data('./database.db')


def optimize_inventory(book_title, data_corrected):
    book_data = data_corrected[data_corrected['Title'] == book_title]
    book_data['Loan Date'] = pd.to_datetime(book_data['Loan Date'])
    book_data['YearMonth'] = book_data['Loan Date'].dt.to_period('M')
    monthly_book_loan_counts = book_data['YearMonth'].value_counts(
    ).sort_index()
    monthly_book_loan_counts.index = monthly_book_loan_counts.index.to_timestamp()

    if len(monthly_book_loan_counts) < 2:
        print(f"Not enough data points for forecasting {book_title}.")
        return

    model = ExponentialSmoothing(
        monthly_book_loan_counts, trend='add', seasonal=None, seasonal_periods=None)
    fit = model.fit()
    forecast = fit.forecast(6)

    service_level_factor = 1.65
    historical_std = monthly_book_loan_counts.std()
    safety_stock = service_level_factor * historical_std
    lead_time_demand = forecast.iloc[0]
    reorder_point = lead_time_demand + safety_stock
    order_quantity = lead_time_demand

    inventory_optimization_results[book_title] = {
        'Reorder Point': reorder_point,
        'Order Quantity': order_quantity,
        'Forecasted Demand': forecast.tolist(),
        'Safety Stock': safety_stock
    }


# Optimización de inventario para los 10 libros más demandados
top_books_list = df['Title'].value_counts().head(10).index
global inventory_optimization_results
inventory_optimization_results = {}
for book_title in top_books_list:
    optimize_inventory(book_title, df)

# Extraer los datos relevantes para la visualización
titles = []
reorder_points = []
order_quantities = []
safety_stocks = []

for book, data in inventory_optimization_results.items():
    titles.append(book)
    reorder_points.append(data['Reorder Point'])
    order_quantities.append(data['Order Quantity'])
    safety_stocks.append(data['Safety Stock'])

inventory_df = pd.DataFrame({
    'Book Title': titles,
    'Reorder Point': reorder_points,
    'Order Quantity': order_quantities,
    'Safety Stock': safety_stocks
})

# Agrupar por biblioteca y clasificación, y contar el número de préstamos
library_classification = df.groupby(
    ['Library Name', 'Clasification']).size().reset_index(name='Loan Count')

# Visualización de optimización de inventario
st.subheader("Optimización de Inventario para los 10 libros más demandados")
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.bar(inventory_df['Book Title'], inventory_df['Reorder Point'],
        color='b', alpha=0.7, label='Reorder Point')
ax1.set_xlabel('Título del Libro')
ax1.set_ylabel('Punto de Reorden', color='b')
ax1.tick_params(axis='y', labelcolor='b')
plt.xticks(rotation=90)

ax2 = ax1.twinx()
ax2.plot(inventory_df['Book Title'], inventory_df['Order Quantity'],
         color='r', marker='o', linestyle='--', label='Order Quantity')
ax2.set_ylabel('Cantidad de Pedido', color='r')
ax2.tick_params(axis='y', labelcolor='r')

fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Optimización de Inventario para los 10 Libros Más Demandados')
plt.grid(True)
st.pyplot(fig)

# Visualización de tipos de libros más rentados por biblioteca
fig = px.bar(library_classification, x='Library Name', y='Loan Count',
             color='Clasification', title='Tipos de Libros más Rentados por Biblioteca')
fig.update_layout(
    xaxis_title="Nombre de la Biblioteca",
    yaxis_title="Cantidad de Préstamos",
    title="Tipos de Libros más Rentados por Biblioteca"
)
st.plotly_chart(fig)

# Visualización de libros rentados en época ordinaria vs. exámenes
loan_counts_exam_vs_ordinary = df.groupby(
    'Is Exam Period').size().reset_index(name='Loan Count')
loan_counts_exam_vs_ordinary['Within Exam Week'] = loan_counts_exam_vs_ordinary['Is Exam Period'].apply(
    lambda x: 'Con exámenes' if x == 1 else 'Sin exámenes')

color_normal = 'blue'
color_examenes = 'red'

fig2 = go.Figure()

# Adding bars for both periods
fig2.add_trace(go.Bar(
    x=loan_counts_exam_vs_ordinary['Within Exam Week'],
    y=loan_counts_exam_vs_ordinary['Loan Count'],
    marker_color=[color_normal if period ==
                  'Sin exámenes' else color_examenes for period in loan_counts_exam_vs_ordinary['Within Exam Week']],
    text=loan_counts_exam_vs_ordinary['Loan Count'],
    textposition='outside'
))

fig2.update_layout(
    title='Libros rentados en época ordinaria vs. exámenes',
    xaxis_title='Periodo',
    yaxis_title='Total de préstamos',
    showlegend=False
)

st.plotly_chart(fig2)

# Group by library and exam period, and count the number of loans
loan_counts_by_library_and_period = df.groupby(
    ['Library Name', 'Is Exam Period']).size().reset_index(name='Loan Count')
loan_counts_by_library_and_period['Within Exam Week'] = loan_counts_by_library_and_period['Is Exam Period'].apply(
    lambda x: 'Con exámenes' if x == 1 else 'Sin exámenes')

fig3 = go.Figure()

# Adding bars for each library and period
for period, color in zip(['Sin exámenes', 'Con exámenes'], [color_normal, color_examenes]):
    period_data = loan_counts_by_library_and_period[
        loan_counts_by_library_and_period['Within Exam Week'] == period]
    fig3.add_trace(go.Bar(
        x=period_data['Library Name'],
        y=period_data['Loan Count'],
        name=period,
        marker_color=color,
        text=period_data['Loan Count'],
        textposition='inside'
    ))

# Update layout for better readability
fig3.update_layout(
    title='Libros rentados en época ordinaria vs. exámenes por biblioteca',
    xaxis_title='Biblioteca',
    yaxis_title='Total de préstamos',
    barmode='group',
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    legend_title='Period',
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

# Adding annotations in the center of the bars
for data in fig3.data:
    data.insidetextanchor = 'middle'
    data.textfont = dict(color='white')

st.plotly_chart(fig3)

# Group by classification and count the number of loans
loan_counts_by_topic = df.groupby(
    'Clasification').size().reset_index(name='Loan Count')

# Create bar chart with different colors for each bar using Plotly
fig4 = go.Figure()

fig4.add_trace(go.Bar(
    x=loan_counts_by_topic['Clasification'],
    y=loan_counts_by_topic['Loan Count'],
    text=loan_counts_by_topic['Loan Count'],
    textposition='outside'
))

# Update layout for better readability
fig4.update_layout(
    title='Préstamos por Clasificación',
    xaxis_title='Clasificación',
    yaxis_title='Préstamos por Clasificación',
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

# Adding annotations on top of the bars
for data in fig4.data:
    data.insidetextanchor = 'end'
    data.textfont = dict(color='black')

st.plotly_chart(fig4)

# Explicación del Análisis
st.markdown("""
## Explicación del Análisis

### Optimización de Inventario de Libros
El objetivo de este análisis es optimizar el inventario de los libros más demandados en la biblioteca. Se utilizan técnicas de modelado de series temporales y estadísticas para calcular el punto de reorden y la cantidad de pedido para cada libro, asegurando que siempre haya suficiente stock para satisfacer la demanda.

- **Punto de Reorden (Reorder Point)**: Es el nivel de inventario en el que se debe realizar un nuevo pedido para evitar la falta de stock.
- **Cantidad de Pedido (Order Quantity)**: Es la cantidad de libros que se debe pedir cada vez que se alcanza el punto de reorden.
- **Stock de Seguridad (Safety Stock)**: Es una cantidad adicional de inventario para cubrir variaciones inesperadas en la demanda.

### Tipos de Libros más Rentados por Biblioteca
Esta visualización muestra la cantidad de préstamos de libros clasificados por género y por biblioteca. Esto ayuda a identificar qué tipos de libros son más populares en cada biblioteca, facilitando la gestión del inventario y la adquisición de nuevos libros en función de la demanda específica.

- **Eje X (Nombre de la Biblioteca)**: Representa las distintas bibliotecas.
- **Eje Y (Cantidad de Préstamos)**: Representa el número total de préstamos de libros en cada biblioteca.

### Libros rentados en época ordinaria vs. exámenes
Este gráfico compara la cantidad de libros rentados en periodos ordinarios y en periodos de exámenes. Permite analizar cómo varía la demanda de libros durante los exámenes y ajustar la gestión del inventario en consecuencia.

- **Periodo**: Diferencia entre periodos ordinarios y periodos de exámenes.
- **Total de préstamos**: Cantidad total de préstamos de libros en cada periodo.
""")
