# Vehicle Sales Data Analysis

## Project Overview
This project provides a comprehensive, end-to-end analysis of vehicle sales data, focusing on data cleaning, feature engineering, exploratory data analysis (EDA), market and seller analysis, clustering, time series analysis, and price prediction modeling. The project leverages Python (pandas, scikit-learn, TensorFlow, matplotlib, seaborn, statsmodels) and Power BI to uncover insights into car pricing, market segmentation, and seller strategies, offering valuable information for stakeholders in the automotive industry.

## Directory Structure
```
Vehicle_sales_data_analysis/
├── Data/                         # Raw and processed datasets
│   ├── car_prices_ori.csv        # Original dataset
│   ├── car_prices_clean.csv      # Cleaned dataset
│   └── decomposed_car_sales_resample_period.csv # Time series decomposition output
├── Data_Classification/          # Data classification scripts and outputs
│   ├── data_classification.py
│   └── sorted_data/              # Classified CSVs by column value
├── Img/                          # Output images from analyses
│   ├── Cluster_Analysis_Img/
│   ├── EDA_img/
│   ├── Feature_Engieering_And_Selection_Img/
│   ├── Market_Analysis_Img/
│   ├── Price_Prediction_Model_Img/
│   ├── Seller_Analysis_Img/
│   └── Time_Series_Analysis_Img/
├── Power BI/                     # Power BI dashboard and report
│   ├── Vehicle_sales_data_analysis.pbix
│   └── Vehicle_sales_data_analysis.pdf
├── Price_Prediction_Model/       # Model training, usage, and scalers
│   ├── price_prediction_model_traning.py
│   ├── price_prediction_model_usage.py
│   └── scaler/
├── Clustering_Analysis.py
├── Data_cleaning_and_preprocess.py
├── Descriptive_statistics.py
├── EDA.py
├── Feature_Engieering_And_Selection.py
├── Market_Analysis.py
├── Seller_Analysis.py
├── Time_Series_Analysis.py
├── README.md
├── LICENSE
└── Vehicle Sales Data Analysis.pdf
```

## Data Description
- **car_prices_ori.csv**: The original dataset, sourced from Kaggle's Vehicle Sales and Market Trends Dataset.
- **car_prices_clean.csv**: The cleaned dataset after preprocessing.
- **decomposed_car_sales_resample_period.csv**: Output from time series decomposition.

## Detailed Script Descriptions

### 1. Data Cleaning and Preprocessing (`Data_cleaning_and_preprocess.py`)
- Loads the original dataset and performs:
  - Initial statistics (attribute count, names, data count)
  - Handling missing values (drop or impute)
  - Outlier identification and (optionally) removal
  - Data type correction
  - Duplicate removal
- Outputs: `car_prices_clean.csv` (cleaned data)

### 2. Exploratory Data Analysis (`EDA.py`)
- Visualizes missing values, distributions, pairwise relationships, and correlations
- Analyzes frequency of categorical variables and cross-tabulations
- Outputs: Multiple plots in `Img/EDA_img/` (e.g., heatmaps, histograms, pairplots, boxplots)

### 3. Descriptive Statistics (`Descriptive_statistics.py`)
- Prints summary statistics for numeric and categorical columns

### 4. Feature Engineering and Selection (`Feature_Engieering_And_Selection.py`)
- Creates new features (e.g., car age, mileage per year)
- Standardizes features, splits data, and builds a TensorFlow regression model
- Computes and visualizes feature importances
- Outputs: Feature importance plot in `Img/Feature_Engieering_And_Selection_Img/`

### 5. Market Analysis (`Market_Analysis.py`)
- Analyzes average selling price by make/model and the impact of condition
- Visualizes results with barplots, boxplots, and scatterplots
- Outputs: Plots in `Img/Market_Analysis_Img/`

### 6. Seller Analysis (`Seller_Analysis.py`)
- Evaluates seller performance by average price and sales volume
- Combines metrics in a dual-axis plot
- Outputs: Plots in `Img/Seller_Analysis_Img/`

### 7. Clustering Analysis (`Clustering_Analysis.py`)
- Standardizes selected features and applies K-means clustering
- Optimizes cluster count and visualizes clusters
- Outputs: Plots in `Img/Cluster_Analysis_Img/`

### 8. Time Series Analysis (`Time_Series_Analysis.py`)
- Aggregates sales by time period (day, week, month, year)
- Decomposes time series into trend, seasonality, and residuals
- Outputs: Plots in `Img/Time_Series_Analysis_Img/` and decomposition CSV in `Data/`

### 9. Price Prediction Model
- **Training (`Price_Prediction_Model/price_prediction_model_traning.py`)**:
  - Preprocesses data, splits into train/test, builds and trains a TensorFlow regression model
  - Saves model and scalers to `Price_Prediction_Model/scaler/`
- **Usage (`Price_Prediction_Model/price_prediction_model_usage.py`)**:
  - Loads the trained model and scalers
  - Preprocesses new data and predicts selling prices

### 10. Data Classification (`Data_Classification/data_classification.py`)
- Interactive script to select a column and value, then exports filtered data to CSV in `Data_Classification/sorted_data/`

### 11. Power BI Dashboard
- **Vehicle_sales_data_analysis.pbix**: Interactive dashboard for visual exploration
- **Vehicle_sales_data_analysis.pdf**: Exported report for sharing insights

## Outputs
- **Images**: All analysis and model outputs are saved as PNGs in the respective `Img/` subfolders.
- **Models and Scalers**: Saved in `Price_Prediction_Model/scaler/` for reuse.
- **Classified Data**: Saved in `Data_Classification/sorted_data/`.
- **Power BI**: Dashboard and PDF report in `Power BI/`.

## Usage
- All scripts are intended for educational and informational purposes.
- To run analyses, execute the relevant Python scripts in order (starting with data cleaning).
- For model inference, use the provided usage script with new data formatted as in the training set.
- For interactive classification, run the classification script and follow prompts.
- Power BI dashboard can be opened with Power BI Desktop.

## License
This repository is provided for viewing and educational purposes only. No license is applied; all rights reserved. For permissions, contact the author.

## Author
Tsz Fong Chan

## Acknowledgments
Special thanks to Syed Anwar Afridi for providing the dataset used in this analysis.

For any inquiries or further information, please feel free to reach out.
