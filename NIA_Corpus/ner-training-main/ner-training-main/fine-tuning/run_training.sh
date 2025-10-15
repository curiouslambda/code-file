export HUGGING_FACE_HUB_TOKEN="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"
export WANDB_PROJECT=ner_train
export CUDA_VISIBLE_DEVICES=3,2,1,0

RUN_NAME="arspraxia_ner_indobert-base_1214_jsonl_filtered_ent-lvl"
python train.py \
    --model_name_or_path "indobenchmark/indobert-base-p2" \
    --dataset_name "curiouslambda/ner_train" \
    --num_data_proc 35 \
    --max_seq_length 128 \
    --return_entity_level_metrics True \
    --output_dir "../../$RUN_NAME/" \
    --do_train --do_eval --do_predict \
    --ignore_mismatched_sizes False \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 2000 \
    --save_steps 2000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --weight_decay 0.01 \
    --load_best_model_at_end \
    --seed 42 --data_seed 42 \
    --run_name "$RUN_NAME" \
    --metric_for_best_model "overall_f1" \
    --greater_is_better True \
    --report_to "wandb" \
    --fp16 True