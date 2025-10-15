import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_directory(dir_):
    # This function will be executed by each process in the pool
    jsonl_files = [file for file in dir_.glob('*.jsonl')]
    if len(jsonl_files) == 0:
        if 'json' in str(dir_):
            json_files = [json.loads(file.read_text(encoding='utf-8-sig')) for file in dir_.glob('*.json') if 'twitter' in str(file)]
            combined_filepath = dir_ / 'combined_twitter.jsonl'
            combined_filepath.write_text('\n'.join(json.dumps(file) for file in json_files), encoding='utf-8-sig')

            input_list = [json.loads(file.read_text(encoding='utf-8-sig')) for file in dir_.glob('*.json') if 'twitter' not in str(file)]
            sublist_size = 3
            json_files = [input_list[i:i + sublist_size] for i in range(0, len(input_list), sublist_size)]
            for i, files in enumerate(json_files):
                combined_filepath = dir_ / f'combined_{str(i)}.jsonl'
                combined_filepath.write_text('\n'.join(json.dumps(file) for file in files), encoding='utf-8-sig')
        
        else:
            json_files = [json.loads(file.read_text(encoding='utf-8-sig')) for file in dir_.glob('*.json')]
            combined_filepath = dir_ / 'combined.jsonl'
            combined_filepath.write_text('\n'.join(json.dumps(file) for file in json_files), encoding='utf-8-sig')


if __name__ == "__main__":
    parent_path = Path('../../')
    directories = [dir_ for dir_ in parent_path.iterdir() if dir_.is_dir()]

    # Using ProcessPoolExecutor to execute processes in parallel
    with ProcessPoolExecutor() as executor:
        # Submit all the directories to the executor
        futures = {executor.submit(process_directory, dir_): dir_ for dir_ in directories}
        
        # Progress bar for completed tasks
        for future in tqdm(as_completed(futures), total=len(directories), desc="Combine json files"):
            pass
