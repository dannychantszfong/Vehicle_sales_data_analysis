import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# Function to perform time series analysis
def perform_time_series_analysis(resample_period='M'):
    
    # Load the dataset

    file_path = r"Data\car_prices_clean.csv"
    #LOCAL_PATH = os.path.dirname(__file__)  # Get local file path
    #file_path = os.path.join(LOCAL_PATH, 'car_sales_data.csv')  # Ensure your data file is named appropriately
    data = pd.read_csv(file_path)

    # Ensure the 'saledate' column is in datetime format
    data['saledate'] = pd.to_datetime(data['saledate'], errors='coerce')

    # Print the datatypes
    print(data.dtypes)

    # Check for null values after conversion
    print("Number of null values in 'saledate' column after conversion:", data['saledate'].isnull().sum())

    # Drop rows with null values in 'saledate' column
    data = data.dropna(subset=['saledate'])

    # Remove any timezone information by converting to naive datetime
    data['saledate'] = data['saledate'].apply(lambda x: x.replace(tzinfo=None))

    # Print the first few rows to verify the conversion
    print(data['saledate'].head())

    # Set the 'saledate' column as the index
    data.set_index('saledate', inplace=True)

    # Ensure the index is a DatetimeIndex
    print("Index type:", type(data.index))

    # Print the range of dates
    print("Date range:", data.index.min(), "to", data.index.max())

    # Aggregate data to get sales based on the resample period
    sales_data = data.resample(resample_period).size()

    # Check the number of observations
    print(f"Number of observations (resample period: {resample_period}):", len(sales_data))

    # Plot the sales data
    plt.figure(figsize=(10, 6))
    plt.plot(sales_data, label=f'Car Sales ({resample_period})')
    plt.title(f'Car Sales Over Time ({resample_period})')
    plt.xlabel('Date')
    plt.ylabel('Number of Cars Sold')
    plt.legend()
    plt.savefig(f"Img\Time_Series_Analysis_Img\Car Sales Over Time ({resample_period}.png")
    plt.show()

    # Check if there are enough observations for seasonal decomposition
    min_observations = {
        'D': 365,
        'W': 2 * 52,
        'M': 2 * 12,
        'Y': 2
    }.get(resample_period, 24)

    if len(sales_data) >= min_observations:
        # Decompose the time series
        decomposition = seasonal_decompose(sales_data, model='additive')

        # Plot the decomposed components
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        ax1.set_title(f'Decomposition of Car Sales ({resample_period})')
        decomposition.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        plt.tight_layout()
        plt.savefig(r"Img\Time_Series_Analysis_Img\\time_series_analysis.png")
        plt.show()

        # Optional: Save the decomposed components to CSV files
        decomposition_df = pd.DataFrame({
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        })
        decomposition_df.to_csv('Data\decomposed_car_sales_{resample_period}.csv')

        print(f"Time series decomposition complete. Decomposed components saved to 'decomposed_car_sales_{resample_period}.csv'.")
    else:
        print(f"Not enough observations for seasonal decomposition for resample period '{resample_period}'. At least {min_observations} are required.")

# Call the function with different resample periods
perform_time_series_analysis('D')  # Daily
perform_time_series_analysis('W')  # Weekly
perform_time_series_analysis('M')  # Monthly
perform_time_series_analysis('Y')  # Yearly
