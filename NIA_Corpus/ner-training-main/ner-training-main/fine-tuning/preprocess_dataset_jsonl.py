import datasets
from pathlib import Path
from datasets import VerificationMode
from tqdm import tqdm


if __name__ == "__main__":
    parent_path = Path('../../')
    file_paths = [str(file) for dir_ in tqdm(parent_path.iterdir(), desc='Reading directories', total=209) if dir_.is_dir() for file in dir_.glob('*.jsonl')]

    print("Reading files...")
    dataset = datasets.load_dataset(
        "json", 
        data_files=file_paths,
        encoding="utf-8",
        num_proc=45,
        verification_mode=VerificationMode.NO_CHECKS
    )

    train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    train_test_valid_dataset = datasets.DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})

    print('Sample:')
    for item in train_test_valid_dataset['train']:
        print(item)
        break
    
    
    print('Uploading to huggingface')
    train_test_valid_dataset.push_to_hub("curiouslambda/ner_train", private=True)
