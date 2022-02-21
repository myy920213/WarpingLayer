python train.py \
--val-shot 1 \
--save-path './experiments/miniImageNet/protonet' \
--gpu 0 \
--train-shot 5 \
--network ProtoNet \
--head ProtoNet \
--dataset miniImageNet \
--episodes-per-batch 8 \
#--eps 0.2
