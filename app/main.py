import streamlit as st
import pandas as pd
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM library_loans"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def preprocess_data(df):
    df['Loan Date'] = pd.to_datetime(df['Loan Date'], errors='coerce')
    if df['Loan Date'].isnull().any():
        st.sidebar.write(
            "There are invalid or missing dates in the 'Loan Date' column. These rows will be dropped.")
        df = df.dropna(subset=['Loan Date'])
    df['Loan Date'] = df['Loan Date'].dt.date
    loan_counts = df.groupby(
        ['Loan Date']).size().reset_index(name='Loan Count')
    return loan_counts


def is_exam_period(date, exam_periods):
    for start, end in exam_periods:
        if start <= pd.to_datetime(date) <= end:
            return 1
    return 0


def add_features(loan_counts, exam_periods):
    loan_counts['Is Exam Period'] = loan_counts['Loan Date'].apply(
        lambda date: is_exam_period(date, exam_periods))
    loan_counts['Day of Week'] = loan_counts['Loan Date'].apply(
        lambda date: pd.to_datetime(date).dayofweek)
    loan_counts['Month'] = loan_counts['Loan Date'].apply(
        lambda date: pd.to_datetime(date).month)
    loan_counts.set_index('Loan Date', inplace=True)
    return loan_counts


def train_model(loan_counts):
    # Fit SARIMA model
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)  # Weekly seasonality
    exog = loan_counts[['Is Exam Period', 'Day of Week', 'Month']]
    model = SARIMAX(loan_counts['Loan Count'],
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog)
    results = model.fit(disp=False)
    st.sidebar.write(f"Model AIC: {results.aic}")
    return results


def predict_future_loans(model, exam_periods, start_date, end_date):
    future_dates = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'Loan Date': future_dates})
    future_df['Is Exam Period'] = future_df['Loan Date'].apply(
        lambda date: is_exam_period(date, exam_periods))
    future_df['Day of Week'] = future_df['Loan Date'].apply(
        lambda date: pd.to_datetime(date).dayofweek)
    future_df['Month'] = future_df['Loan Date'].apply(
        lambda date: pd.to_datetime(date).month)
    future_df.set_index('Loan Date', inplace=True)

    # Forecast future loan counts
    forecast = model.get_forecast(steps=len(future_df), exog=future_df[[
                                  'Is Exam Period', 'Day of Week', 'Month']])
    future_df['Predicted Loan Count'] = forecast.predicted_mean
    return future_df


def plot_predictions(future_df):
    plt.figure(figsize=(10, 6))
    plt.plot(future_df.index, future_df['Predicted Loan Count'], color='red')
    plt.xlabel('Date')
    plt.ylabel('Predicted Loan Count')
    plt.title('Predicted Loan Counts Over Time')
    plt.xticks(rotation=45)
    st.pyplot(plt)


def main():
    st.title("Library Loan Prediction")
    st.sidebar.header("Settings")

    # Load data
    df = load_data('./database.db')

    # Display the data
    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Data from SQLite Database")
        st.write(df)

    # Preprocess data
    loan_counts = preprocess_data(df)

    # Display the aggregated data
    if st.sidebar.checkbox("Show aggregated loan counts"):
        st.subheader("Aggregated Loan Counts")
        st.write(loan_counts)

    # Define exam periods
    exam_periods = [
        (pd.to_datetime('2023-05-01'), pd.to_datetime('2023-05-31')),
        (pd.to_datetime('2023-12-01'), pd.to_datetime('2023-12-31'))
    ]

    # Add features to data
    loan_counts = add_features(loan_counts, exam_periods)

    # Train the model
    model = train_model(loan_counts)

    # Predict future loan counts
    future_df = predict_future_loans(
        model, exam_periods, '2024-01-01', '2024-12-31')

    # Display predictions
    st.subheader("Predicted Loan Counts for Future Dates")
    st.write(future_df)

    # Plot the predictions
    plot_predictions(future_df)


if __name__ == "__main__":
    main()
