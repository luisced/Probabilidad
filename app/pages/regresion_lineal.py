
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from connection import load_data

# Load data
df = load_data('./database.db')

# Ensure 'Loan Date' is in datetime format
df['Loan Date'] = pd.to_datetime(df['Loan Date'])

# Add week information
df['Week'] = df['Loan Date'].dt.isocalendar().week

# Calculate weekly loan counts
df['Within Exam Week'] = df['Is Exam Period'] == 1
weekly_loans = df.groupby(['Week', 'Within Exam Week']
                          ).size().reset_index(name='Loan Count')

# Separate into exam weeks and non-exam weeks
exam_weeks = weekly_loans[weekly_loans['Within Exam Week'] == True]
non_exam_weeks = weekly_loans[weekly_loans['Within Exam Week'] == False]

# Linear regression function


def linear_regression_plot(x, y, label, color):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    return model.coef_[0], model.intercept_, y_pred


# All weeks regression
coef_all, intercept_all, y_pred_all = linear_regression_plot(
    weekly_loans['Week'], weekly_loans['Loan Count'], 'Préstamos por semana', 'blue')
# Non-exam weeks regression
coef_non_exam, intercept_non_exam, y_pred_non_exam = linear_regression_plot(
    non_exam_weeks['Week'], non_exam_weeks['Loan Count'], 'Préstamos en semanas sin exámenes', 'green')
# Exam weeks regression
coef_exam, intercept_exam, y_pred_exam = linear_regression_plot(
    exam_weeks['Week'], exam_weeks['Loan Count'], 'Préstamos en semanas con exámenes', 'red')

# January to July
jan_to_jul = df[(df['Loan Date'].dt.month >= 1) &
                (df['Loan Date'].dt.month <= 6)]
weekly_loans_jan_to_jul = jan_to_jul.groupby(
    'Week').size().reset_index(name='Loan Count')
coef_jan_to_jul, intercept_jan_to_jul, y_pred_jan_to_jul = linear_regression_plot(
    weekly_loans_jan_to_jul['Week'], weekly_loans_jan_to_jul['Loan Count'], 'Enero a Junio', 'purple')

# August to December
aug_to_dec = df[(df['Loan Date'].dt.month >= 8) &
                (df['Loan Date'].dt.month <= 12)]
weekly_loans_aug_to_dec = aug_to_dec.groupby(
    'Week').size().reset_index(name='Loan Count')
coef_aug_to_dec, intercept_aug_to_dec, y_pred_aug_to_dec = linear_regression_plot(
    weekly_loans_aug_to_dec['Week'], weekly_loans_aug_to_dec['Loan Count'], 'Agosto a Diciembre', 'orange')

# Cumulative loans
weekly_loans_jan_to_jul['Cumulative Loans'] = weekly_loans_jan_to_jul['Loan Count'].cumsum()
weekly_loans_aug_to_dec['Cumulative Loans'] = weekly_loans_aug_to_dec['Loan Count'].cumsum()

# Plot all weeks
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=weekly_loans['Week'], y=weekly_loans['Loan Count'], mode='markers', name='Loan Count'))
fig1.add_trace(go.Scatter(
    x=weekly_loans['Week'], y=y_pred_all, mode='lines', name='Linear Fit'))
fig1.update_layout(title='Regresión lineal de préstamos por semana',
                   xaxis_title='Semana', yaxis_title='Número de préstamos')

# Plot non-exam weeks
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=non_exam_weeks['Week'], y=non_exam_weeks['Loan Count'],
               mode='markers', name='Loan Count', marker=dict(color='green')))
fig2.add_trace(go.Scatter(x=non_exam_weeks['Week'], y=y_pred_non_exam,
               mode='lines', name='Linear Fit', line=dict(color='green')))
fig2.update_layout(title='Regresión lineal de préstamos por semana sin exámenes',
                   xaxis_title='Semana', yaxis_title='Número de préstamos')

# Plot exam weeks
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=exam_weeks['Week'], y=exam_weeks['Loan Count'],
               mode='markers', name='Loan Count', marker=dict(color='red')))
fig3.add_trace(go.Scatter(x=exam_weeks['Week'], y=y_pred_exam,
               mode='lines', name='Linear Fit', line=dict(color='red')))
fig3.update_layout(title='Regresión lineal de préstamos por semana con exámenes',
                   xaxis_title='Semana', yaxis_title='Número de préstamos')

# Plot January to July
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=weekly_loans_jan_to_jul['Week'], y=weekly_loans_jan_to_jul['Loan Count'],
               mode='markers', name='Loan Count', marker=dict(color='purple')))
fig4.add_trace(go.Scatter(
    x=weekly_loans_jan_to_jul['Week'], y=y_pred_jan_to_jul, mode='lines', name='Linear Fit', line=dict(color='purple')))
fig4.update_layout(title='Regresión lineal con préstamos semanales semestre 1232',
                   xaxis_title='Semana', yaxis_title='Número de préstamos')

# Plot August to December
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=weekly_loans_aug_to_dec['Week'], y=weekly_loans_aug_to_dec['Loan Count'],
               mode='markers', name='Loan Count', marker=dict(color='orange')))
fig5.add_trace(go.Scatter(
    x=weekly_loans_aug_to_dec['Week'], y=y_pred_aug_to_dec, mode='lines', name='Linear Fit', line=dict(color='orange')))
fig5.update_layout(title='Regresión lineal con préstamos semanales semestre 1238',
                   xaxis_title='Semana', yaxis_title='Número de préstamos')

# Cumulative loans plot
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=weekly_loans_jan_to_jul['Week'], y=weekly_loans_jan_to_jul[
               'Cumulative Loans'], mode='lines', name='January to July', line=dict(color='blue')))
fig6.add_trace(go.Scatter(x=weekly_loans_aug_to_dec['Week'], y=weekly_loans_aug_to_dec['Cumulative Loans'],
               mode='lines', name='August to December', line=dict(color='orange')))
fig6.update_layout(title='Cumulative Number of Loans Per Semester',
                   xaxis_title='Week Number', yaxis_title='Cumulative Number of Loans')

# Streamlit layout

# Streamlit layout
st.title("Análisis de Préstamos de Libros")

st.subheader("Regresión lineal de préstamos por semana")
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

st.subheader("Regresión lineal de préstamos semanales por semestre")
st.plotly_chart(fig4)
st.plotly_chart(fig5)

st.subheader("Número acumulativo de préstamos por semestre")
st.plotly_chart(fig6)

# Report on the regressions
st.markdown("""
## Informe de Análisis de Regresión Lineal

### Regresión Lineal de Préstamos por Semana

#### Todas las Semanas
- **Coeficiente (β1):** -1.99
- **Intercepto (β0):** 240.60

La fórmula de la regresión lineal para todas las semanas es:
\[ \text{Préstamos} = -1.99 \times \text{Semana} + 240.60 \]

Esto indica que en promedio, los préstamos disminuyen en aproximadamente 2 libros por cada semana que pasa, comenzando desde un valor inicial de 240.60 préstamos.

#### Semanas sin Exámenes
- **Coeficiente (β1):** -2.02
- **Intercepto (β0):** 261.04

La fórmula de la regresión lineal para las semanas sin exámenes es:
\[ \text{Préstamos} = -2.02 \times \text{Semana} + 261.04 \]

Esto sugiere que en las semanas sin exámenes, los préstamos disminuyen en promedio en 2.02 libros por semana, comenzando desde 261.04 préstamos.

#### Semanas con Exámenes
- **Coeficiente (β1):** -1.04
- **Intercepto (β0):** 180.19

La fórmula de la regresión lineal para las semanas con exámenes es:
\[ \text{Préstamos} = -1.04 \times \text{Semana} + 180.19 \]

Esto indica que en las semanas con exámenes, los préstamos disminuyen en promedio en 1.04 libros por semana, comenzando desde 180.19 préstamos.

### Regresión Lineal de Préstamos Semanales por Semestre

#### Enero a Junio
- **Coeficiente (β1):** 0.47
- **Intercepto (β0):** 243.01

La fórmula de la regresión lineal para el período de enero a junio es:
\[ \text{Préstamos} = 0.47 \times \text{Semana} + 243.01 \]

Esto sugiere que durante el primer semestre del año, los préstamos aumentan ligeramente en 0.47 libros por semana, comenzando desde 243.01 préstamos.

#### Agosto a Diciembre
- **Coeficiente (β1):** -6.43
- **Intercepto (β0):** 457.96

La fórmula de la regresión lineal para el período de agosto a diciembre es:
\[ \text{Préstamos} = -6.43 \times \text{Semana} + 457.96 \]

Esto indica que durante el segundo semestre del año, los préstamos disminuyen significativamente en 6.43 libros por semana, comenzando desde 457.96 préstamos.

### Interpretación de Resultados
Los resultados de las regresiones lineales muestran patrones interesantes en el comportamiento de los préstamos de libros:

1. **Disminución General:** En general, los préstamos de libros tienden a disminuir a lo largo del tiempo en todas las semanas, con una disminución más pronunciada en las semanas sin exámenes comparadas con las semanas con exámenes.
2. **Efecto de los Exámenes:** Durante las semanas con exámenes, aunque los préstamos también disminuyen, lo hacen a un ritmo más lento comparado con las semanas sin exámenes, lo que puede indicar que los estudiantes siguen necesitando libros aunque estén en periodo de exámenes.
3. **Variación Semestral:**
    - **Enero a Junio:** Los préstamos tienden a aumentar ligeramente durante el primer semestre del año.
    - **Agosto a Diciembre:** Los préstamos disminuyen de manera más pronunciada durante el segundo semestre, lo que podría estar relacionado con la estructura del año académico y las vacaciones de fin de año.

Esta información puede ser útil para planificar la gestión del inventario y las estrategias de adquisición de libros a lo largo del año, asegurando que la biblioteca pueda satisfacer la demanda de los usuarios de manera más efectiva.
""")
