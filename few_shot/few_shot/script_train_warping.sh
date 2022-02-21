python train_opt.py \
--dataset miniImageNet \
--pre_train './experiments/miniImageNet/protonet/best_model.pth' \
--gamma 50.0 \
--num_batch 300 \
--val-shot 1 \
--save-path './experiments/miniImageNet/protonet+W' \
--gpu 0 \
--cpu 0 \
--train-shot 5 \
--network ProtoNet \
--head ProtoNet \
--a_clip_max 0.000625 \
--a_clip_min 0.000625 \
--val-episode 500  \


#--lr 0.01 \
#--episodes-per-batch 8 \
#--print-every 20 \
#--lr-epoch '2 4 8' \
#--lr-val '1.0 0.06 0.012 0.0024'



