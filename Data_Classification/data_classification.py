import pandas as pd

# Load your dataset (assuming it's a CSV file, modify the path as necessary)
file_path = r'vehicle_sales_data_analysis_n_visualisitaion\car_sales_data_analysis\cleaned_data.csv'
df = pd.read_csv(file_path)

def select_column(df):
    """Function to allow the user to select a column."""
    while True:
        try:
            print("\nColumns available:")
            for idx, col in enumerate(df.columns):
                print(f"{idx}: {col}")

            col_num = int(input("\nEnter the number of the column you want to select: "))

            if 0 <= col_num < len(df.columns):
                selected_column = df.columns[col_num]
                print(f"\nYou have selected the column: {selected_column}")
                column_details(df[selected_column])
                break
            else:
                print("Invalid column number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def column_details(column_data):
    """Function to display details of the selected column."""
    count_dict = column_data.value_counts().to_dict()
    select_detail(count_dict, column_data)

def select_detail(count_dict, column_data):
    """Function to allow the user to select a key from the column details."""
    print("\nDetails available:")
    temp_dict = {i+1: key for i, key in enumerate(count_dict)}

    for key, value in temp_dict.items():
        print(f"{key}: {value}")

    while True:
        try:
            selected_key = int(input("\nEnter the number of the key you want to select: "))
            selected_value = temp_dict[selected_key]
            print(f"\nYou have selected the key: {selected_value}")
            save_to_csv(selected_value, column_data)
            break
        except (ValueError, KeyError):
            print("Invalid key. Please try again.")

def save_to_csv(selected_key, column_data):
    """Function to save related data to a new CSV file."""
    related_data = df[df[column_data.name] == selected_key]
    file_name = f"{selected_key}.csv"
    related_data.to_csv(file_name, index=False)
    print(f"\nThe related data has been saved to {file_name}")

# Call the function to select a column
select_column(df)
