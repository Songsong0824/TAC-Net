#!/bin/bash

python3 main_my_model.py \
--do_train \
--epochs=100 \
--lr 5e-5 \
--gpu_id '0' \
--seed 1 \
--audio1_num_hidden_layers 6 \
--audio2_num_hidden_layers 6 \
--audio3_num_hidden_layers 6 \
--audio1_dim 74 \
--audio2_dim 74 \
--audio3_dim 74 \
--binary_threshold 0.25 \
--recon_mse_weight 1.0 \
--aug_mse_weight 1.0 \
--beta_mse_weight 0.0 \
--lsr_clf_weight 0.01 \
--recon_clf_weight 0.0 \
--aug_clf_weight 0.1 \
--shuffle_aug_clf_weight 0.1 \
--total_aug_clf_weight 1.0 \
--cl_weight 1.0 \
--aligned \
--dataset CMU-Mosei
