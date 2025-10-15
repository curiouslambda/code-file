import json
from pathlib import Path
from tqdm import tqdm
from utils import label2id_orig as label2id

def process_jsonl_files(directory):
    # Create a Path object for the directory
    dir_path = Path(directory)

    # Process each file in the directory that ends with .jsonl
    for file_path in dir_path.glob('*.jsonl'):
        # Read the content of the JSONL file
        with file_path.open('r', encoding='utf-8-sig') as file:
            lines = file.readlines()

        # Reorganize the content
        reorganized_data = []
        for line in lines:
            try:
                data = json.loads(line)
                for item in data['data']:
                    reorganized_data.append(item)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file {file_path.name}: {line}")

        # Write the reorganized content back to the same file
        with file_path.open('w', encoding='utf-8') as file:
            for entry in reorganized_data:
                file.write(json.dumps(entry) + '\n')

def process_jsonl_files_clean(directory):
    # Create a Path object for the directory
    dir_path = Path(directory)

    # Process each file in the directory that ends with .jsonl
    for file_path in dir_path.glob('*.jsonl'):
        # Read the content of the JSONL file
        with file_path.open('r', encoding='utf-8-sig') as file:
            lines = file.readlines()

        # Reorganize the content
        reorganized_data = []
        for line in lines:
            try:
                data = json.loads(line)
                if 'ner_tags' in data:
                    reorganized_data.append({
                        'tokens': data['tokens'],
                        'ner_tags': data['ner_tags'],
                    })
                else:
                    try:
                        reorganized_data.append({
                            "tokens": data['Raw_data'].split(),
                            "ner_tags": [label2id[tag.strip()] if tag.strip() in label2id else label2id[tag.split('/')[-1].strip()] for tag in data['Entities_list']],
                        })
                    except AttributeError:
                        print('Attribute error for', data)
                    except KeyError:
                        print('KeyError for', data)

            except json.JSONDecodeError:
                print(f"Error decoding JSON in file {file_path.name}: {line}")

        # Write the reorganized content back to the same file
        with file_path.open('w', encoding='utf-8') as file:
            for entry in reorganized_data:
                file.write(json.dumps(entry) + '\n')

def process_jsonl_files_len(directory):
    # Create a Path object for the directory
    dir_path = Path(directory)

    # Process each file in the directory that ends with .jsonl
    for file_path in dir_path.glob('*.jsonl'):
        # Read the content of the JSONL file
        with file_path.open('r', encoding='utf-8-sig') as file:
            lines = file.readlines()

        # Reorganize the content
        reorganized_data = []
        for line in lines:
            try:
                data = json.loads(line)
                if len(data['tokens']) == len(data['ner_tags']):
                    reorganized_data.append({
                        'tokens': data['tokens'],
                        'ner_tags': data['ner_tags'],
                    })
                # else:
                #     print('Not equal for', data)

            except json.JSONDecodeError:
                print(f"Error decoding JSON in file {file_path.name}: {line}")

        # Write the reorganized content back to the same file
        with file_path.open('w', encoding='utf-8') as file:
            for entry in reorganized_data:
                file.write(json.dumps(entry) + '\n')

# Replace '/path/to/jsonl/files' with the path to the directory containing your JSONL files
# process_jsonl_files('/mnt/nas2/kikiputri/ner/data/sample_48')
parent_path = Path('/mnt/nas2/kikiputri/ner/data')
directories = [str(dir_) for dir_ in tqdm(parent_path.iterdir(), desc='Open dir') if dir_.is_dir()]
for dir_ in tqdm(directories, desc="Reformat jsonl"):
    process_jsonl_files_len(dir_)
