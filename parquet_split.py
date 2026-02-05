import argparse
import os
from datasets import Dataset

def split_parquet(input_path, output_path, num_rows):
    """
    从Parquet文件中提取指定数量的数据并保存到新文件。
    
    Args:
        input_path (str): 输入Parquet文件路径
        output_path (str): 输出Parquet文件路径
        num_rows (int): 要提取的行数
    """
    print(f"正在读取文件: {input_path}")
    
    try:
        # 加载数据集
        dataset = Dataset.from_parquet(input_path)
        total_rows = len(dataset)
        print(f"原始数据集总行数: {total_rows}")
        
        # 检查请求的行数是否超过总行数
        if num_rows > total_rows:
            print(f"警告: 请求的行数 ({num_rows}) 超过了总行数 ({total_rows})。将导出所有数据。")
            num_rows = total_rows
            
        # 提取前 num_rows 条数据
        # 使用 select 进行切片，保持数据类型一致
        small_dataset = dataset.select(range(num_rows))
        
        print(f"正在保存前 {num_rows} 条数据到: {output_path}")
        
        # 保存为 Parquet 文件
        small_dataset.to_parquet(output_path)
        
        print("保存成功！")
        print("=" * 30)
        print(f"新文件信息:")
        print(f"路径: {output_path}")
        print(f"行数: {len(small_dataset)}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取Parquet文件的前N条数据用于快速测试")
    
    # 定义命令行参数
    parser.add_argument("--input", "-i", type=str, default="data/pro/train.parquet", help="输入Parquet文件路径 (默认: data/pro/train.parquet)")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出Parquet文件路径 (默认: [文件名]_top[N].parquet)")
    parser.add_argument("--num", "-n", type=int, default=10000, help="要提取的数据条数 (默认: 10000)")
    
    args = parser.parse_args()
    
    input_path = args.input
    
    # 如果未指定输出路径，自动生成
    if not args.output:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_top{args.num}{ext}"
    else:
        output_path = args.output
        
    split_parquet(input_path, output_path, args.num)
