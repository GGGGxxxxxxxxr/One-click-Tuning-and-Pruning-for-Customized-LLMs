import json

with open("/Users/leilu/Desktop/ATO_llm/nlp_dataset_collections/InternalMed_Harrison.txt", 'r') as file:
    lines = [line.strip() for line in file if line.strip()]

# Get the first 300 lines
lines_300 = lines[:300]
# Prepare data in the required format
data = [{'text': line} for line in lines_300]

# Write data to a JSON Lines file
with open('perplexity_dataset.jsonl', 'w') as outfile:
    for entry in data:
        json.dump(entry, outfile)
        outfile.write('\n')