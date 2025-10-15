# NER Fine-tuning

This repository consists of scripts to train and doing the inference of the NER model.


## Before Training: Environment setup
Install all dependencies using conda. Note that the `yml` file is exist in the root directory of this project.
```
conda env create -f environment.yml
```

## Before Training: Data Preprocessing
Before we run the training, we need to preprocess the data. The goal is to make the data format to be the same as the default NER data format in huggingface. Basically, the data should consists of list of `tokens` and `ner_tags` (similar to [CONLL-2003 dataset](https://huggingface.co/datasets/conll2003)).

The data preprocessing is consist of 3 steps.
1. Modify the json files format.

   In this step, we split the tokens and modify the NER labels so that it consists of NER tag IDs instead of the NER tag labels. To do this, please run the following code:
   ```
   python preprocess_dataset_json.py
   ```

   Please make sure that you have a correct data path in this line of code:
   ```
   parent_path = Path(# insert the data path here)
   ```

   Also, please note that this code will replace the content of the original json file. So please have the file backup in case of errors.

2. Combine json files into one jsonl file per directory.

   Since we process millions of data, it will be faster if we read one jsonl file per directory instead of millions of file. So, we combine it using this code:
   ```
   python merge_json.py
   ```

   For twitter data, the jsonl file will be a combination of 3 json files. For news data, the jsonl file will be a combination of all json files.

   Again, please make sure that you have change the data path in this line of code:
   ```
   parent_path = Path(# insert the data path here)
   ```

3. Read jsonl file, split the data into train, validation, and test.

   In this step, we will split the data and the upload the data to huggingface repository for faster data reading time. 

   Before running the code, please make sure you have modify the path and the data filename in the `preprocess_dataset_jsonl.py`.

   Modify the data path in this line of code:
   ```
   parent_path = Path(# insert the data path here)
   ```

   Modify the data repository name in this line of code:
   ```
   train_test_valid_dataset.push_to_hub("rifkiaputri/arspraxia_ner", private=True)
   ```
   In the above code example, the data will be uploaded under "rifkiaputri" huggingface account with "arspraxia_ner" as the dataset name. The dataset will also be private since we set `private=True`. By setting the dataset type as private, it means that you have to access the dataset using a private huggingface token key. Therefore, please make sure that you have the access token before uploading. Consider to use your own account if you have one.

   After modifying the above lines, we can run the code:
   ```
   python preprocess_dataset_jsonl.py
   ```

## Training
After the pre-processing has finished, you can run the training by simply run this script:
```
bash run_training.sh
```

Note that before running you have to make sure that you set the parameters inside the script file correctly, including the huggingface tokens, wandb project name, data training name (`dataset_name`), and output path of the model (`output_dir`). You also need to modify the `CUDA_VISIBLE_DEVICES` parameter according to the GPUs that are available.

The training log will be saved in [wandb](https://wandb.ai/). Please make sure you have make the wandb account. 


## Inference
Below is the steps that you need to follow to do the model prediction/inference.

### 1. Environment Setup
Basically, there are two core libraries that you need to install before you can load the model. The libraries are Pytorch and Huggingface. Using conda for virtual environment is also recommended.

First, create an environment with a specific version of Python (3.8 is recommended). Then, activate the environment.
```
conda create -n ner python=3.8
conda activate ner
```

Then, install Pytorch (GPU version).
```
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Finally, install HuggingFace.
```
pip install transformers==4.33.1
```

### 2. Model Download
You can download the trained model [here](https://drive.google.com/file/d/1lsln21VlhntG_OGSOZmgfao9iOps_lUV/view?usp=sharing). Then, compressed the zip file.

### 3. Model Load
(Note: you can see the complete example code in `inference.ipynb` file.)

First, you can set the GPU device that you want use.
```
!export CUDA_VISIBLE_DEVICES=1
```

Import necessary package.
```
from transformers import pipeline
```

Set the model path with the path of the model that you have downloaded in the Step 2 (Model Download). Then, load the model using `pipeline` class from HuggingFace.
```
model_path = # set the model path here
classifier = pipeline("ner", model=model_path, aggregation_strategy="simple")
```

Then, you can do the inference by calling the `classifier` object.
```
# The input_text can be in a form of string or list
input_text = "Drama ' High School Love On ' telah merilis beberapa potongan gambar yang menampilkan ChoA Crayon Pop dalam balutan seragam sekolah ."
classifier(input_text)
```
