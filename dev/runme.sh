#!/bin/bash

# ========================================================
# Workspace, dataset, and evaluation tools
# ========================================================
# Modify the following paths to your own workspace
WORKSPACE="/rhome/beyer/reproduction/PM2S"
# Modify the following paths to your own dataset directory
ASAP="$WORKSPACE/pm2s/data/asap-dataset-master"
A_MAPS="$WORKSPACE/pm2s/data/A-MAPS_1"
CPM="$WORKSPACE/pm2s/data/ClassicalPianoMIDI-dataset-main"
ACPAS="$WORKSPACE/pm2s/data/ACPAS-dataset"

# # ========================================================
# # Feature preparation
# # ========================================================
# python3 feature_preparation.py \
#     --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
#     --feature_folder $WORKSPACE/features \
#     --workers 4 \


# # ========================================================
# # Model training
# # ========================================================
# # feature can be 'beat', 'quantisation', 'time_signature', 'key_signature', 'hand_part'
python3 train.py \
    --workspace $WORKSPACE \
    --ASAP $ASAP \
    --A_MAPS $A_MAPS \
    --CPM $CPM \
    --feature 'beat' \
    # --full_train


# # ========================================================
# # Save model state dict
# # ========================================================
# # Change the model_checkpoint_path to your own trained model checkpoint path, this will save the model to the default path (replacing the pre-trained model state dict)
# python3 save_model.py \
#     --model_checkpoint_path /import/c4dm-05/ll307/workspace/PM2S-draft/mlruns/5/9cd46d4570eb45ca8e72052b479e7ac4/checkpoints/epoch=57-val_loss=0.23-val_f1=0.86.ckpt \
#     --feature 'key_signature' \
#     # --beat_model_checkpoint ../_model_state_dicts/beat/RNNJointBeatModel_fullTrain.pth
