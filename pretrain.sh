export SOURCE=/mnt/d/M3/Projects/BCB/crisprBert/
export OUTPUT_PATH=$SOURCE/Models/crisprBert/
export CONFIG_PATH=$OUTPUT_PATH/config/

python crisprBert.py \
    --data_path $SOURCE/unlabeled_sgrna_fixed.txt \
    --config_path $CONFIG_PATH \
    --output_path CONFIG_PATH \
    --make_configs \
    --vocab_size 52_000 \
   	--max_position_embeddings 514 \
    --num_attention_heads 12 \
    --num_hidden_layers 6 \
    --type_vocab_size 1 \
    --block_size 128 \
    --mlm \
    --mlm_probability 0.025 \
    --overwrite_output_dir \
    --num_train_epochs 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 6 \
    --learning_rate 4e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --logging_dir "Models/crisprBert/log.txt" \
    --logging_steps 10_00 \
    --eval_steps 10_000 \
    --save_steps 10_000 \
    --metric_for_best_model "eval_loss" \
	--save_total_limit 10 \
    # --prediction_loss_only \
    --do_train \
    --do_eval \
    --logging_first_step \
    # --logging_nan_inf_filter \
	# --greater_is_better \
	--load_best_model_at_end \