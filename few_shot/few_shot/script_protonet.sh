python train_opt.py \
--pre_train './experiments/miniImageNet/protonet_5_n/best_model.pth' \
--gamma 1.0 \
--num_batch 200 \
--val-shot 1 \
--save-path './experiments/miniImageNet/protonet_opt_5_shot_1.0_200_0.000625' \
--gpu 0 \
--cpu 0 \
--train-shot 5 \
--network ProtoNet \
--head ProtoNet \
--a_clip_max 0.000625 \
--a_clip_min 0.000625