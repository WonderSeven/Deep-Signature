echo '++++++++++++++++++++++++++++++++++++++++++++++++'
echo '+++++         FrameLSTM-Head: EGFR         +++++'
echo '++++++++++++++++++++++++++++++++++++++++++++++++'

gpu_ids='5'

for seed in {0..4}
do
  echo "Running exp with $seed"
  python main.py \
  --gpu_ids $gpu_ids \
  --data_root '/hdd2/qtx/Datasets/GPCR-2' \
  --data_name 'GPCR' \
  --data_reg 'gpcr' \
  --atom_type 'backbone' \
  --traj_nums 26 \
  --groups 5 \
  --folds 5 \
  --input_dim 3 \
  --algorithm FrameLSTM \
  --local_mode 'head_and_tail' \
  --num_clusters 50 \
  --epochs 200 \
  --batch_size 16 \
  --mode 'train' \
  --lr 1e-3 \
  --weight_decay 5e-4 \
  --loss_func 'bce' \
  --seed $seed \
  --save_path './logs/GPCR_B_5G_5F/FrameLSTM_head_tail_C50' \
  --record
  echo "=================="
done
