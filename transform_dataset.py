import pandas as pd
import json

def excel_to_jsonl_without_file_column(excel_file, jsonl_file):
    # 读取 Excel 文件中的第一个 sheet
    df = pd.read_excel(excel_file)

    # 移除不需要的 'File' 列（如果存在）
    if 'File' in df.columns:
        df = df.drop(columns=['File'])

    # 将 DataFrame 转换为字典列表，每一行数据变为一个 JSON 对象
    data = df.to_dict(orient='records')

    # 将数据逐行写入 JSONL 文件
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Successfully converted {excel_file} to {jsonl_file}, with 'File' column removed.")

# 示例用法
excel_to_jsonl_without_file_column('/Users/leilu/Desktop/ATO_llm/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx', 'HQS_train.jsonl')