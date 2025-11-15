
model_name=ASE_TS

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --hidden_size 512 \
  --batch_size 128 \
  --learning_rate 0.02 \
  --des 'Exp' \
  --itr 1
