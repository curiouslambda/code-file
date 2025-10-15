from pathlib import Path
from utils import label2id_orig as label2id
from tqdm import tqdm
import json


if __name__ == "__main__":
    parent_path = Path('../../')
    file_paths = [str(file) for dir_ in tqdm(parent_path.iterdir(), desc='Reading directories', total=209) if dir_.is_dir() for file in dir_.glob('*.json')]
    
    # for dir_ in parent_path.iterdir(): print(dir_)
    # print('\n\n---------------')
    # print(file_paths)

    # for file_path in tqdm(file_paths, desc='Change key format'):
    #     with open(file_path, 'r', encoding='utf-8-sig') as f:
    #         data = json.load(f)
        

    #     if 'Raw_data' not in data['data']:
    #         continue

    #     data['data'] = [{
    #         'tokens': line['Raw_data'].encode("utf-8").decode("utf-8").split(),
    #         'ner_tags': [label2id[tag] for tag in line['Entities_list']]
    #     } for line in data['data']]
        
        # with open(file_path, 'w', encoding='utf-8') as f:
        #     json.dump(data, f)

    for file_path in tqdm(file_paths, desc='Change key format'):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        for line in data['data']:
            if 'Raw_data' not in line:
                continue

            data['data'] = [{
                'tokens': line['Raw_data'].encode("utf-8").decode("utf-8").split(),
                'ner_tags': [label2id[tag] for tag in line['Entities_list']]
            }]

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)

    # print(data['data'])