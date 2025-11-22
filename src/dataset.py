import pandas as pd
import os

def load_and_prepare_data(data_path='final.csv'):
    """
    Loads the Shakespeare parallel corpus from a CSV file.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with 'modern' and 'shakespearean' columns.
    """
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return None

    # Load the CSV file using pandas
    df = pd.read_csv(data_path)

    # Rename columns for convenience (if necessary)
    # Assuming original column names are 'Modern English' and 'Shakespeare English'
    df.rename(columns={
        'Modern English': 'modern',
        'Shakespeare English': 'shakespearean'
    }, inplace=True)

    print("Successfully loaded the dataset.")
    print(f"Total number of sentence pairs: {len(df)}")

    # Print the first 5 rows for verification
    print("\nData Head:")
    print(df.head())

    # Check for missing values
    print("\nMissing values check:")
    print(df.isnull().sum())

    return df

# --- Call from the main script ---
if __name__ == '__main__':
    dataframe = load_and_prepare_data()