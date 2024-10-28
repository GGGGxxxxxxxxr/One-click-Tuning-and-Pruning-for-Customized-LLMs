import json

def subtract_jsonl_by_question(a_jsonl, b_jsonl, output_jsonl):
    # 将 b.jsonl 中的所有 QUESTION 加载到一个集合
    with open(b_jsonl, 'r', encoding='utf-8') as b_file:
        b_questions = {json.loads(line)["QUESTION"] for line in b_file}

    # 打开 a.jsonl，并过滤掉 QUESTION 在 b.jsonl 中的条目
    with open(a_jsonl, 'r', encoding='utf-8') as a_file:
        remaining_data = [
            line for line in a_file
            if json.loads(line)["QUESTION"] not in b_questions
        ]

    # 将剩余数据写入新的 JSONL 文件
    with open(output_jsonl, 'w', encoding='utf-8') as output_file:
        output_file.writelines(remaining_data)

    print(f"Successfully saved remaining data to {output_jsonl}")

# 示例用法
subtract_jsonl_by_question('/Users/leilu/Desktop/ATO_llm/nlp_dataset_collections/PubMedQA/pubMedQA_test.jsonl', 'newnew_test.jsonl', 'remaining_data.jsonl')