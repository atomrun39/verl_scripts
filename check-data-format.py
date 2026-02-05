import pandas as pd
import sys

# 请将此处替换为你实际的 parquet 文件路径
parquet_path = "path/to/your/train.parquet" 

try:
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows.")
    print(f"Columns found: {df.columns.tolist()}")
    
    if 'prompt' in df.columns:
        print("✅ Column 'prompt' exists.")
        # 检查是否所有行都有 prompt
        null_count = df['prompt'].isnull().sum()
        if null_count > 0:
            print(f"⚠️ Warning: Found {null_count} rows with null prompt!")
        else:
            print("✅ All rows have prompt data.")
            print("Sample data:", df.iloc[0]['prompt'])
    else:
        print("❌ ERROR: Column 'prompt' NOT found!")
        # 模糊匹配检查
        for col in df.columns:
            if 'prompt' in col.lower():
                print(f"   Did you mean: '{col}'?")

except Exception as e:
    print(f"Error reading file: {e}")