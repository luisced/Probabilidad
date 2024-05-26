import streamlit as st
import pandas as pd
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM library_loans"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


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
    if st.sidebar.checkbox("Show plot"):
        plot_predictions(future_df)

    # Additional analyses and plots
    if st.sidebar.checkbox("Show loan counts by library location"):
        library_loan_counts = df[df['Library Name'] !=
                                 'Unknown']['Library Name'].value_counts()
        st.subheader("Loan Counts by Library Location")
        st.bar_chart(library_loan_counts)

    if st.sidebar.checkbox("Show loan counts by book title and library location"):
        top_books_list = df['Title'].value_counts().head(10).index
        top_books_data = df[df['Title'].isin(
            top_books_list) & (df['Library Name'] != 'Unknown')]
        library_book_counts = top_books_data.groupby(
            ['Title', 'Library Name']).size().unstack(fill_value=0)
        st.subheader(
            "Loan Counts of Top 10 Most Demanded Books by Library Location")
        st.write(library_book_counts)
        st.bar_chart(library_book_counts)

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

    if st.sidebar.checkbox("Show inventory optimization results"):
        st.subheader("Inventory Optimization for Top 10 Most Demanded Books")
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


if __name__ == "__main__":
    main()
