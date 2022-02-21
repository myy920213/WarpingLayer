#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.13 -ad [100,200,250] --gpu '0,1' --epochs 300 --lr 1e-1 --br -t 3 --gamma 100 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_300.pth.tar';
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.13 -ad [100,200,250] --gpu '2,3' --epochs 300 --lr 1e-2 --br -t 4 --gamma 100 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_400.pth.tar';

#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [100,200,250] --gpu '0,1' --epochs 300 --lr 1e-2 --br -t 7 --gamma 1000 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_300.pth.tar' --contin 1;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 8 --gamma 1000 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_400.pth.tar' --contin 1;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 9 --gamma 1000;

#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 10 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_400.pth.tar' --contin 1;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 11 --gamma 500 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_400.pth.tar' --contin 1;

#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 3 --gamma 10 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_400.pth.tar' --contin 1;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 4 --gamma 100 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_400.pth.tar' --contin 1;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.25 -ad [400,500,550] --gpu '0,1,2,3' --epochs 700 --br -t 101;
 
#python main_shot_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [500,600,650] --gpu '0,1' --epochs 700 --br -t 1 --gamma 100 --resume 'basepath/Cifar100-SHOT-VAE/parameter/train_time_1/checkpoint.pth.tar' --contin 1;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '0,1,2,3' --epochs 700 --br -t 12;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 11 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_12/checkpoint_400.pth.tar' --contin 1;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 7 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_12/checkpoint_400.pth.tar' --contin 1
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 12 --gamma 100 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_12/checkpoint_400.pth.tar' --contin 1;


#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '0,1,2,3' --epochs 700 --br -t 500 --ewm 1e-1;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '0,1,2,3' --epochs 700 --br -t 501 --ewm 1e-2;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '0,1,2,3' --epochs 700 --br -t 502 --ewm 5e-3;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '0,1,2,3' --epochs 700 --br -t 503 --ewm 1e-4;


#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 500 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_500/checkpoint_400.pth.tar' --contin 1 --ewm 1e-1;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 501 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_501/checkpoint_400.pth.tar' --contin 1 --ewm 1e-2;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 502 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_502/checkpoint_400.pth.tar' --contin 1 --ewm 5e-3;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 503 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_503/checkpoint_400.pth.tar' --contin 1 --ewm 1e-4;

#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 503 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_503/checkpoint_400.pth.tar' --contin 1 --ewm 1e-4;

#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 600 --gamma 100 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_500/checkpoint_400.pth.tar' --contin 1 --ewm 1e-1;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 601 --gamma 100 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_501/checkpoint_400.pth.tar' --contin 1 --ewm 1e-2;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 602 --gamma 100 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_502/checkpoint_400.pth.tar' --contin 1 --ewm 5e-3;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 503 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_503/checkpoint_400.pth.tar' --contin 1 --ewm 1e-4;


#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 500 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_500/checkpoint_400.pth.tar' --contin 1 --ewm 1e-1;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 501 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_501/checkpoint_400.pth.tar' --contin 1 --ewm 1e-2;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 502 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_502/checkpoint_400.pth.tar' --contin 1 --ewm 5e-3;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 504 --ewm 5e-4;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 504 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_504/checkpoint_400.pth.tar' --contin 1 --ewm 5e-4;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 504 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_504/checkpoint_400.pth.tar' --contin 1 --ewm 5e-4;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.25 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 102 --ewm 5e-3;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.25 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 103 --ewm 1e-2;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.05 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 104;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 105;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.05 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 106 --ewm 5e-4;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 107 --ewm 5e-3;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.05 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 108 --ewm 8e-4;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 109 --ewm 2e-3;

#python main_shot_vae.py -bp basepath --net-name wideresnet-28-2 --dataset "Cifar100" --annotated-ratio 0.15 -ad [500,600,650] --epochs 700 --br -t 16 --ewm 5e-3;

#python main_shot_vae.py -bp basepath --net-name wideresnet-28-2 --dataset "Cifar100" --annotated-ratio 0.10 -ad [500,600,650] --gpu '2,3' --epochs 700 --br -t 18 --ewm 2e-3;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 200;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 504 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_200/checkpoint_400.pth.tar' --contin 1;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 204;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 204 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_204/checkpoint_400.pth.tar' --contin 1;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 205;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 205 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_205/checkpoint_400.pth.tar' --contin 1;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.12 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 206;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 206 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_205/checkpoint_400.pth.tar' --contin 1;

#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 300 --gamma 100 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_10/checkpoint_400.pth.tar' --contin 1

#python main_classifier.py -bp basepath --net-name wideresnet-28-2 --gpu '0,1' -b 128 --dataset "Cifar100" --annotated-ratio 0.05 -t 100;
#python main_classifier.py -bp basepath --net-name wideresnet-28-2 --gpu '2,3' -b 128 --dataset "Cifar100" --annotated-ratio 0.15 -t 101;
#python main_classifier.py -bp basepath --net-name wideresnet-28-2 --gpu '0,1' -b 128 --dataset "Cifar100" --annotated-ratio 0.25 -t 102;

#python main_classifier_opt.py -bp basepath --net-name wideresnet-28-2 --gpu '2,3' -b 128 --dataset "Cifar100" --annotated-ratio 0.05 -t 100 --gamma 100 --resume 'basepath/Cifar100-SSL-Classifier/parameter/train_time:100/checkpoint.pth.tar';
#python main_classifier_opt.py -bp basepath --net-name wideresnet-28-2 --gpu '0,1' -b 128 --dataset "Cifar100" --annotated-ratio 0.15 -t 101 --gamma 100 --resume 'basepath/Cifar100-SSL-Classifier/parameter/train_time:101/checkpoint.pth.tar';
#python main_classifier_opt.py -bp basepath --net-name wideresnet-28-2 --gpu '0,1' -b 128 --dataset "Cifar100" --annotated-ratio 0.25 -t 102 --gamma 100 --resume 'basepath/Cifar100-SSL-Classifier/parameter/train_time:102/checkpoint.pth.tar';


#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.05 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 600 --ewm 1e-1;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.05 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 601 --ewm 1e-2;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.05 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 602 --ewm 5e-3;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.05 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 603 --ewm 5e-4;

#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 600 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_600/checkpoint_400.pth.tar' --contin 1 --ewm 1e-1;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 600 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_600/checkpoint_400.pth.tar' --contin 1 --ewm 1e-1;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 601 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_601/checkpoint_400.pth.tar' --contin 1 --ewm 1e-2;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 601 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_601/checkpoint_400.pth.tar' --contin 1 --ewm 1e-2;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 602 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_602/checkpoint_400.pth.tar' --contin 1 --ewm 5e-3;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 602 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_602/checkpoint_400.pth.tar' --contin 1 --ewm 5e-3;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.05 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 604 --ewm 1e-3;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 604 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_604/checkpoint_400.pth.tar' --contin 1 --ewm 1e-3;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 604 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_604/checkpoint_400.pth.tar' --contin 1 --ewm 1e-3;
#python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 603 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_603/checkpoint_400.pth.tar' --contin 1 --ewm 5e-4;
#python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.07 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 603 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_603/checkpoint_400.pth.tar' --contin 1 --ewm 5e-4;

#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.15 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 705 --ewm 1e-4;
#python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.25 -ad [400,500,550] --gpu '2,3' --epochs 700 --br -t 805 --ewm 1e-4;




python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.27 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 801 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_801/checkpoint_400.pth.tar' --contin 1 --ewm 1e-2;
python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.27 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 801 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_801/checkpoint_400.pth.tar' --contin 1 --ewm 1e-2;
python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.27 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 802 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_802/checkpoint_400.pth.tar' --contin 1 --ewm 5e-3;
python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.27 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 802 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_802/checkpoint_400.pth.tar' --contin 1 --ewm 5e-3;
python main_M2_vae.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.25 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 804 --ewm 1e-3;
python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.27 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 804 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_804/checkpoint_400.pth.tar' --contin 1 --ewm 1e-3;
python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.27 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 804 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_804/checkpoint_400.pth.tar' --contin 1 --ewm 1e-3;
python main_M2_vae_opt.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.27 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 803 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_803/checkpoint_400.pth.tar' --contin 1 --ewm 5e-4;
python main_M2_vae_opt_ab.py -bp basepath --net-name wideresnet-28-2 -b 256 --dataset "Cifar100" --annotated-ratio 0.27 -ad [400,500,550] --gpu '0,1' --epochs 700 --br -t 803 --gamma 200 --resume 'basepath/Cifar100-M2-VAE/parameter/train_time_803/checkpoint_400.pth.tar' --contin 1 --ewm 5e-4;








