import scipy.stats as stats
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils import is_exam_period
from connection import load_data


df = load_data('./database.db')

# Split the data into exam periods and non-exam periods
exam_data = df[df['Is Exam Period'] == True]
non_exam_data = df[df['Is Exam Period'] == False]

print(exam_data.head())
print(non_exam_data.head())
# Perform t-test
t_stat, p_value = stats.ttest_ind(
    exam_data['Loan Time Seconds'], non_exam_data['Loan Time Seconds'], equal_var=False)

# Display t-test results
st.subheader("Prueba de Hipótesis")
st.write(f"Estadístico t: {t_stat:.4f}")
st.write(f"p-valor: {p_value:.4f}")

if p_value < 0.05:
    st.write("Rechazamos la hipótesis nula: Hay una diferencia significativa en la demanda de libros durante los periodos de exámenes.")
else:
    st.write("No podemos rechazar la hipótesis nula: No hay una diferencia significativa en la demanda de libros durante los periodos de exámenes.")

# Plot the box plot
# Plot the box plot
st.subheader("Distribución de los Tiempos de Préstamo")
fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = [exam_data['Loan Time Seconds'],
                non_exam_data['Loan Time Seconds']]
ax.boxplot(data_to_plot, labels=['Periodo de Exámenes', 'No Exámenes'])
ax.set_xlabel('Periodo')
ax.set_ylabel('Tiempo de Préstamo en Segundos')
ax.set_title(
    'Distribución de los Tiempos de Préstamo en Periodos de Exámenes y No Exámenes')
st.pyplot(fig)


confidence_level = 0.95
# confidence interval for the mean loan time during exam periods
exam_time_mean = df['Loan Time Seconds'].mean()
exam_time_sem = stats.sem(df['Loan Time Seconds'])
exam_time_ci = stats.t.interval(confidence_level, len(
    df['Loan Time Seconds'])-1, loc=exam_time_mean, scale=exam_time_sem)

st.write(f"Intervalo de Confianza para la Media del Tiempo de Préstamo en Periodos de Exámenes: {
         exam_time_ci}")

# confidence interval for the mean loan time outside exam periods
non_df = df[df['Is Exam Period'] == False]
non_exam_time_mean = non_df['Loan Time Seconds'].mean()
non_exam_time_sem = stats.sem(non_df['Loan Time Seconds'])
non_exam_time_ci = stats.t.interval(confidence_level, len(
    non_df['Loan Time Seconds'])-1, loc=non_exam_time_mean, scale=non_exam_time_sem)

st.write(f"Intervalo de Confianza para la Media del Tiempo de Préstamo fuera de Periodos de Exámenes: {
         non_exam_time_ci}")
# Cerrar la conexión
