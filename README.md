# Vehicle Sales Data Analysis

## Overview
This project provides a comprehensive, end-to-end analysis of vehicle sales data, focusing on data cleaning, feature engineering, exploratory data analysis (EDA), market and seller analysis, clustering, time series analysis, and price prediction modeling. The project leverages Python (pandas, scikit-learn, TensorFlow, matplotlib, seaborn, statsmodels) and Power BI to uncover insights into car pricing, market segmentation, and seller strategies, offering valuable information for stakeholders in the automotive industry.

## Prerequisites
- Python 3.8 or higher
- pip package manager
- Basic understanding of data analysis and Python programming
- Minimum 8GB RAM recommended for large dataset processing

## Technologies Used
- **Python Libraries**:
  - pandas: Data manipulation and analysis
  - numpy: Numerical computing
  - scikit-learn: Machine learning algorithms
  - matplotlib/seaborn: Data visualization
  - statsmodels: Time series analysis
  - tensorflow: Deep learning and regression modeling
  - joblib: Model serialization
  - jupyter: Interactive development
- **Power BI**: For dashboarding and reporting

## Project Structure
```
Vehicle_sales_data_analysis/
├── Data/                          # Raw and processed data files
│   ├── car_prices_ori.csv        # Original dataset
│   ├── car_prices_clean.csv      # Cleaned dataset
│   └── decomposed_car_sales_resample_period.csv # Time series decomposition output
├── Data_Classification/           # Classification analysis scripts
│   ├── data_classification.py
│   └── sorted_data/              # Classified CSVs by column value
├── Price_Prediction_Model/        # Price prediction model implementation
│   ├── price_prediction_model_traning.py
│   ├── price_prediction_model_usage.py
│   └── scaler/                   # Saved scalers and models
├── Img/                          # Generated visualizations and images
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

## Key Features
- **Data Cleaning and Preprocessing**: 
  - Handling missing values, outliers, and data inconsistencies
  - Data validation and quality checks
  - Standardization of data formats
- **Feature Engineering**: 
  - Creation of new features like car age and mileage per year
  - Feature selection and importance analysis
  - Data transformation and scaling
- **Market Analysis**: 
  - Examination of car prices across different makes and models
  - Impact of vehicle condition on pricing
  - Market trend analysis and seasonality patterns
- **Seller Performance Evaluation**: 
  - Analysis of sales volume and average selling prices
  - Seller segmentation and performance metrics
  - Competitive analysis
- **Advanced Analytics**:
  - Clustering and market segmentation using K-means
  - Time series analysis of sales patterns
  - Predictive modeling for price forecasting

## Installation & Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/dannychantszfong/Vehicle_sales_data_analysis.git
cd Vehicle_sales_data_analysis
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the analysis scripts in the following order:
   - `Data_cleaning_and_preprocess.py`
   - `Feature_Engieering_And_Selection.py`
   - `EDA.py`
   - Other analysis scripts as needed

## Example Outputs
Here are some key insights you can expect from running the analysis:
- Market trend visualizations showing price variations across different car makes
- Seasonal patterns in vehicle sales
- Clustering results showing distinct market segments
- Predictive model performance metrics
- Seller performance rankings and analysis

For detailed visualizations and results, check the `Img/` directory after running the scripts.

## Data Dictionary
Key variables in the dataset include:
- `make`: Car manufacturer
- `model`: Car model name
- `year`: Manufacturing year
- `sellingprice`: Sale price
- `odometer`: Total miles driven
- `condition`: Vehicle condition (numeric scale)
- `mmr`: Manheim Market Report value
- `saledate`: Date of sale
- `trim`, `body`, `transmission`, `state`, `color`, `interior`, `seller`, `vin`: Additional vehicle and transaction details
- [Add more variables as needed]

## Main Scripts Description
- `Data_cleaning_and_preprocess.py`: Initial data cleaning and preprocessing
- `Feature_Engieering_And_Selection.py`: Feature creation and selection
- `EDA.py`: Exploratory Data Analysis
- `Descriptive_statistics.py`: Prints summary statistics for numeric and categorical columns
- `Market_Analysis.py`: Market trend analysis
- `Seller_Analysis.py`: Seller performance analysis
- `Time_Series_Analysis.py`: Temporal pattern analysis
- `Clustering_Analysis.py`: Market segmentation analysis
- `Price_Prediction_Model/price_prediction_model_traning.py`: Model training and saving
- `Price_Prediction_Model/price_prediction_model_usage.py`: Model inference on new data
- `Data_Classification/data_classification.py`: Interactive data classification and export

## Data Sources
The dataset used in this analysis is sourced from Kaggle's Vehicle Sales and Market Trends Dataset, provided by Syed Anwar Afridi.

## Results and Findings
Detailed analysis results can be found in the following locations:
- Visualizations in the `Img/` directory
- Statistical summaries in analysis outputs
- Price prediction models in the `Price_Prediction_Model/` directory
- Classification results in the `Data_Classification/` directory
- Power BI dashboard and PDF report in the `Power BI/` directory

## Outputs
- **Images**: All analysis and model outputs are saved as PNGs in the respective `Img/` subfolders.
- **Models and Scalers**: Saved in `Price_Prediction_Model/scaler/` for reuse.
- **Classified Data**: Saved in `Data_Classification/sorted_data/`.
- **Power BI**: Dashboard and PDF report in `Power BI/`.

## License
This project is licensed for personal and academic use only. Commercial use is not permitted. You are free to:
- Download and use the software
- Modify the code
- Share the software

Under the following conditions:
- You must give appropriate credit by including the original author's name (Danny Chan / Tsz Fong Chan)
- You may not use this software for commercial purposes
- You must include this license notice in any copy or modification of the software

## Acknowledgments
- Special thanks to Syed Anwar Afridi for providing the dataset
- Thanks to all contributors and reviewers

## Contact
- **Author**: Danny Chan (Tsz Fong Chan)
- **Email**: w1819419@my.westminster.ac.uk
- **LinkedIn**: Tsz Fong Chan
- **GitHub**: @dannychantszfong

---
**Note**: This project has been completed. All analyses and implementations are finalized.

## Troubleshooting
Common issues and solutions:
1. **Memory Error**: Try reducing the chunk size in data processing scripts
2. **Missing Dependencies**: Ensure all requirements are installed via `pip install -r requirements.txt`
3. **Data Loading Issues**: Verify the data file is in the correct location and format

## Contributing
While this project is primarily for personal and academic use, suggestions and bug reports are welcome:
1. Fork the repository
2. Create your feature branch
3. Submit a pull request with a clear description of changes
