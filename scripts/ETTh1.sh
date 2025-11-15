
model_name=ASE_TS

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --hidden_size 512 \
  --batch_size 256 \
  --learning_rate 0.02 \
  --des 'Exp' \
  --itr 1
