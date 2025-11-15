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
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return None

    # 使用pandas加载CSV文件
    df = pd.read_csv(data_path)

    # 为了方便，我们可以重命名列名（如果需要的话）
    # 假设原始列名是 'Modern English' 和 'Shakespeare English'
    df.rename(columns={
        'Modern English': 'modern',
        'Shakespeare English': 'shakespearean'
    }, inplace=True)

    print("Successfully loaded the dataset.")
    print(f"Total number of sentence pairs: {len(df)}")

    # 打印前5行数据进行验证
    print("\nData Head:")
    print(df.head())

    # 检查是否有缺失值
    print("\nMissing values check:")
    print(df.isnull().sum())

    return df

# --- 在主脚本中调用 ---
if __name__ == '__main__':
    dataframe = load_and_prepare_data()