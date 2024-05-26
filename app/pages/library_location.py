import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from connection import load_data


df = load_data('./database.db')


# Additional analyses and plots
library_loan_counts = df[df['Library Name'] !=
                         'Unknown']['Library Name'].value_counts()
st.subheader("Loan Counts by Library Location")
st.bar_chart(library_loan_counts)

top_books_list = df['Title'].value_counts().head(10).index
top_books_data = df[df['Title'].isin(
    top_books_list) & (df['Library Name'] != 'Unknown')]
library_book_counts = top_books_data.groupby(
    ['Title', 'Library Name']).size().unstack(fill_value=0)
st.subheader(
    "Loan Counts of Top 10 Most Demanded Books by Library Location")
st.write(library_book_counts)
st.bar_chart(library_book_counts)
