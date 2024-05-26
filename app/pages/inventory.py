import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from connection import load_data
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title("Optimización de Inventario de Libros")

# Descripción de la funcionalidad
st.markdown("""
## Descripción del Proyecto

**Objetivo:** Optimizar el inventario de los libros más demandados en la biblioteca.

**Metodología:**
- Filtrado y agrupación de datos por título y mes.
- Modelado de series temporales utilizando **Exponential Smoothing**.
- Cálculo del **stock de seguridad** y **punto de reorden**.

**Resultados:** Visualización de los puntos de reorden y cantidades de pedido para los 10 libros más demandados.
""")

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


# Inventory optimization for the top 10 most demanded books
top_books_list = df['Title'].value_counts().head(10).index
global inventory_optimization_results
inventory_optimization_results = {}
for book_title in top_books_list:
    optimize_inventory(book_title, df)

# Extract the relevant data for plotting
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

st.subheader("Optimización de Inventario para los 10 libros más demandados")
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.bar(inventory_df['Book Title'], inventory_df['Reorder Point'],
        color='b', alpha=0.7, label='Reorder Point')
ax1.set_xlabel('Book Title')
ax1.set_ylabel('Reorder Point', color='b')
ax1.tick_params(axis='y', labelcolor='b')
plt.xticks(rotation=90)

ax2 = ax1.twinx()
ax2.plot(inventory_df['Book Title'], inventory_df['Order Quantity'],
         color='r', marker='o', linestyle='--', label='Order Quantity')
ax2.set_ylabel('Order Quantity', color='r')
ax2.tick_params(axis='y', labelcolor='r')

fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Inventory Optimization for Top 10 Most Demanded Books')
plt.grid(True)
st.pyplot(fig)
