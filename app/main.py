import streamlit as st
import pandas as pd
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns
from connection import load_data


def preprocess_data(df):
    df['Loan Date'] = pd.to_datetime(df['Loan Date'], errors='coerce')
    if df.isnull().any().any():
        st.sidebar.write(
            "There are invalid or missing values in the DataFrame. These rows will be dropped.")
        df = df.dropna()
    df['Loan Date'] = df['Loan Date'].dt.date
    loan_counts = df.groupby(
        ['Loan Date']).size().reset_index(name='Loan Count')
    loan_counts['Loan Date'] = pd.to_datetime(loan_counts['Loan Date'])
    loan_counts.set_index('Loan Date', inplace=True)
    return loan_counts


def is_exam_period(date, exam_periods):
    date = pd.to_datetime(date)
    date_day_month = (date.month, date.day)
    for start, end in exam_periods:
        start_day_month = (start.month, start.day)
        end_day_month = (end.month, end.day)
        if start_day_month <= date_day_month <= end_day_month:
            return 1
    return 0


def add_features(loan_counts, exam_periods):
    loan_counts['Is Exam Period'] = loan_counts.index.to_series().apply(
        lambda date: is_exam_period(date, exam_periods))
    loan_counts['Day of Week'] = loan_counts.index.to_series().apply(
        lambda date: date.dayofweek)
    loan_counts['Month'] = loan_counts.index.to_series().apply(
        lambda date: date.month)
    return loan_counts


def train_model(loan_counts):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    exog = loan_counts[['Is Exam Period', 'Day of Week', 'Month']]
    model = SARIMAX(loan_counts['Loan Count'], order=order,
                    seasonal_order=seasonal_order, exog=exog)
    results = model.fit(disp=False)
    st.sidebar.write(f"Model AIC: {results.aic}")
    return results


def predict_future_loans(model, exam_periods, start_date, end_date):
    future_dates = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'Loan Date': future_dates})
    future_df['Is Exam Period'] = future_df['Loan Date'].apply(
        lambda date: is_exam_period(date, exam_periods))
    future_df['Day of Week'] = future_df['Loan Date'].apply(
        lambda date: date.dayofweek)
    future_df['Month'] = future_df['Loan Date'].apply(lambda date: date.month)
    future_df.set_index('Loan Date', inplace=True)

    forecast = model.get_forecast(steps=len(future_df), exog=future_df[[
                                  'Is Exam Period', 'Day of Week', 'Month']])
    future_df['Predicted Loan Count'] = forecast.predicted_mean.values

    # Filter predictions to only include exam periods
    future_df['Predicted Loan Count'] = future_df.apply(
        lambda row: row['Predicted Loan Count'] if row['Is Exam Period'] == 1 else None, axis=1)

    return future_df


def plot_predictions(future_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    future_df['Predicted Loan Count'].plot(kind='line', ax=ax, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Loan Count')
    ax.set_title('Predicted Loan Counts Over Time (Exam Periods Only)')
    plt.xticks(rotation=45)
    st.pyplot(fig)


def main():

    # Set Streamlit page title and sidebar
    st.title("Library Loan Optimization")
    # st.sidebar.header("Settings")

    # Page navigation
    st.page_link("pages/library_location.py",
                 label="Library Location", icon="🏫")
    st.page_link("pages/inventory.py",
                 label="Inventory Optimization", icon="📚")

    # Load data
    df = load_data('./database.db')

    # Display the data
    st.subheader("Data from SQLite Database")
    st.write(df)

    # # Preprocess data
    # loan_counts = preprocess_data(df)

    # # Define exam periods
    # exam_periods = [
    #     (pd.to_datetime('2023-05-01'), pd.to_datetime('2023-05-31')),
    #     (pd.to_datetime('2023-12-01'), pd.to_datetime('2023-12-31'))
    # ]

    # # Add features to data
    # loan_counts = add_features(loan_counts, exam_periods)

    # # Train the model
    # model = train_model(loan_counts)

    # # Predict future loan counts
    # future_df = predict_future_loans(
    #     model, exam_periods, '2024-01-01', '2024-12-31')

    # # Display predictions
    # st.subheader("Predicted Loan Counts for Future Dates")
    # st.write(future_df)

    # # Plot the predictions
    # if st.sidebar.checkbox("Show plot"):
    #     plot_predictions(future_df)


if __name__ == "__main__":
    main()
