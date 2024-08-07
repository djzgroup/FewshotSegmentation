GPU_ID=0

DATASET='scannet'
SPLIT=0
DATA_PATH='../datasets/ScanNet/blocks_bs1_s1'
SAVE_PATH='../log/log_scannet/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20
BASE_WIDTHS='[128, 64]'

PRETRAIN_CHECKPOINT='../log/log_pretrain_scannet/log_pretrain_scannet_S0'
N_WAY=3
K_SHOT=1
N_QUESIES=1
N_TEST_EPISODES=100

NUM_ITERS=40000
EVAL_INTERVAL=4000
LR=0.001
DECAY_STEP=5000
DECAY_RATIO=0.5

args=(--phase 'prototrain' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT"
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR  --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K
      --dgcnn_mlp_widths "$MLP_WIDTHS" --base_widths "$BASE_WIDTHS"  --use_attention
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES
      )

python ../main.py "${args[@]}"