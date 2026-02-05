from datasets import Dataset, load_dataset
import pandas as pd

def read_first_row_with_datasets(file_path):
    """
    使用datasets库读取Parquet文件的第一条数据
    
    Args:
        file_path (str): Parquet文件路径
    """
    try:
        # 方法1: 直接加载整个数据集
        dataset = Dataset.from_parquet(file_path)
        
        # 获取第一条数据
        first_row = dataset[0]
        
        print("使用datasets库读取Parquet文件第一条数据:")
        print("=" * 60)
        for key, value in first_row.items():
            print(f"{key}: {value}")
        
        print(f"\n数据集信息:")
        print(f"总行数: {len(dataset)}")
        print(f"特征: {dataset.features}")
        
        return first_row
        
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    #file_path = "dapo-math-17k_dedup_r1_sys_prompt_mathdapo.parquet"  # 替换为你的Parquet文件路径
    file_path = "data/pro/train.parquet"
    read_first_row_with_datasets(file_path)