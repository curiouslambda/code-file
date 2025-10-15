import logging
import os
import numpy as np
import datasets
import evaluate
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
    Trainer, TrainingArguments, HfArgumentParser, EarlyStoppingCallback,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from arguments import ModelArguments, DataTrainingArguments
from utils import label2id, id2label, label_names
# from indobenchmark import IndoNLGTokenizer
# IndoNLGTokinizer 어딨음?


logger = logging.getLogger(__name__)

def main(label_names):
    # Load arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load tokenizer
    if model_args.model_name_or_path in {"indobenchmark/indogpt", "indobenchmark/indobart-v2"}:
        tokenizer = IndoNLGTokenizer.from_pretrained(
            model_args.model_name_or_path
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(label_names),
            finetuning_task="ner",
        )
        if config.model_type in {"bloom", "gpt2", "roberta"}:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=True,
                add_prefix_space=True,
            )
            if config.model_type == "gpt2":
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=True,
            )

    # Load dataset
    logger.info('*** Dataset loading ***')
    dataset = datasets.load_dataset(data_args.dataset_name)

    logger.info('Casting dataset label...')
    new_features = dataset['train'].features.copy()
    new_features["ner_tags"] = datasets.Sequence(
                                    datasets.features.ClassLabel(
                                        names=label_names
                                    )
                                )
    dataset['train'] = dataset['train'].cast(new_features, num_proc=data_args.num_data_proc)
    dataset['test'] = dataset['test'].cast(new_features, num_proc=data_args.num_data_proc)
    dataset['validation'] = dataset['validation'].cast(new_features, num_proc=data_args.num_data_proc)
    label_names = dataset["train"].features["ner_tags"].feature.names
    label_to_id = {i: i for i in range(len(label_names))}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_names):
        if label.startswith("B-") and label.replace("B-", "I-") in label_names:
            b_to_i_label.append(label_names.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
            is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            # Map tokens to their respective word
            if model_args.model_name_or_path in {"indobenchmark/indogpt", "indobenchmark/indobart-v2"}:
                # Create the word_ids mapping manually because IndoNLGTokenizer doesn't support fast tokenizer
                tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])
                current_word_index = -1
                word_ids = []
                for token in tokens:
                    if token in tokenizer.all_special_tokens:
                        word_ids.append(None)
                    else:
                        if token.startswith("▁"):
                            current_word_index += 1
                        word_ids.append(current_word_index)
            else:
                word_ids = tokenized_inputs.word_ids(batch_index=i)
            
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # Set the special tokens to -100
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    # Assuming we label all tokens
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize dataset
    logger.info('*** Start tokenizing data ***')
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, num_proc=data_args.num_data_proc)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Load evaluator
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }


    # Define model & trainer
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=len(label_names), id2label=id2label, label2id=label2id,
        resume_download=True, ignore_mismatched_sizes=model_args.ignore_mismatched_sizes
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"] if training_args.do_train else None,
        eval_dataset=tokenized_dataset["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate the model on eval dataset
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()
        print(results)
        trainer.log_metrics("eval", results)
        trainer.save_metrics("eval", results)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(tokenized_dataset["test"], metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    # Save the tokenizer and model
    tokenizer.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main(label_names=label_names)
