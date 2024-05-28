import numpy as np
import streamlit as st
from connection import load_data
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title("Caso: Obras San Agustín ")

df = load_data('./database.db')

@st.cache_data
def analize_agustin(df=df, book_title='Obras de San Agustín'):
    st.markdown("""
    **Contexto:**
    Durante la realización de este escrito, se ha observado que la obra de San Agustín es una de las más solicitadas en las bibliotecas. Esto resalta ante los demás datos dento de nuestra población, ya que no se esperaba que este libro tuviera tanta demanda.
                
                
    **Objetivo:**
    - Analizar la demanda de la obra de San Agustín en las bibliotecas.
    - Analziar la distribución de la demanda en el tiempo.
    - Analizar si su demanda es en epocas de examenes o no.
                
    """)

    st.subheader("Desarollo del Análisis")

    ## Filtering data
    st.write("Para realizar el análisis, se ha filtrado la base de datos por el título de la obra de San Agustín y realizar dos analisis: el primero corresponderá a la distribución por biblioteca y el segundo a una distribucióon por mes.")
    st.write("A continuación, se muestra las correspondientes gráficas con los datos filtrados:")

    book_data = df[df['Title'] == book_title]
    book_data['Loan Date'] = pd.to_datetime(book_data['Loan Date'])
    book_data['YearMonth'] = book_data['Loan Date'].dt.to_period('M')
    monthly_book_loan_counts = book_data['YearMonth'].value_counts(
    ).sort_index()
    monthly_book_loan_counts.index = monthly_book_loan_counts.index.to_timestamp()

    ## Plotting 1 - Distribution by Library
    st.subheader("1. Distribución por Biblioteca")

    # Group by libraries and for the book 'Obras de San Agustín', count the number of loans
    book_loan_by_library = book_data['Library Name'].value_counts().reset_index()
    book_loan_by_library.columns = ['Library Name', 'Loan Count']
    
    fig = px.bar(book_loan_by_library, x='Library Name', y='Loan Count', title='Loan Counts by Library')
    st.plotly_chart(fig)

    st.write("Se puede observar que a pesar de la demanda que tiene el libro de 'Obra de San Agustín' este únicamente se encuentra en una de las bibliotecas de la UP. Esto llega a representar un porblema a la hora de su uso, ya que causa que se pierdan más datos de su prestamo, tienendo esta pcoas unidades. No solo causando inconvenientes al momento que sacar estadisticas, sino también para sus estudiantes, ya que yo llegan a cubrir la demanda que tiene.")
    st.write("Como se puede observar en la siguiente gráfica, la demanda por la obra va en un aumento, dando razon a lo concluido anteriormente.")

    ## Plotting 2 - Distribution by Month
    st.subheader("2. Distribución por Mes")
    fig = px.line(monthly_book_loan_counts, title='Loan Counts by Month')
    st.plotly_chart(fig)

    st.write("En la gráfica anterior se puede observar que la demanda de la obra de San Agustín va en aumento, lo que nos lleva a pensar que la demanda de este libro es constante y no depende de la época del año. Esto nos lleva a pensar que la demanda de este libro es constante y no depende de la época del año. Por lo que se recomienda solicitar más copias de la obra en las bibliotecas de la UP.")

    st.write("Siguiendo con el analisis, profundiaremos en la demanda de la obra en epocas de examenes y no examenes. Para ello, se ha realizado un análisis de la demanda de la obra de San Agustín en los meses de examenes y no examenes.")

    ## Plotting 3 - histogram by Exam and Non-Exam Periods
    st.subheader("3. Temporada de Examenes vs No Examenes")
    book_data['Month'] = book_data['Loan Date']
    book_data['Is Exam Period'] = np.where(book_data['Month'].isin([1, 6, 12]), 'Exam Period', 'Non-Exam Period')

    fig = px.histogram(book_data, x='Month', color='Is Exam Period', title='Loan Counts by Month')
    st.plotly_chart(fig)

    st.write("Se observa que la demanda de la obra no tiene una realción directa con los periodos de examenes, sino que es variante a lo largo del año, parece sorprendente que al final de año, la demanda por esta obra aumemnta.")

    st.write("Por último, se ha realizado un análisis de predición sobre la demanda de la obra de San Agustín en la biblioteca de Valencia, sabiendo que esta es la unica biblioteca que tiene la obra. Se busca optimizar la cantidad de copias que se deben solicitar para satisfacer la demanda de la obra.")

    st.subheader("Conclusiones")
    st.write("Con los análisis realizados, se puede concluir que la demanda de la obra de San Agustín es constante a lo largo del año y no depende de los periodos de examenes. Se recomienda solicitar más copias de la obra en las bibliotecas de la UP para satisfacer la demanda de los estudiantes.")  
    
## Mostrar Analisis
analize_agustin()