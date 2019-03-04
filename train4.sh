cd /workspace/mnt/group/test/zhaozhijian/distributed/RefineDetPytorch/
envpy3/bin/python train_dist.py --save_folder ./dist --cfg configs/refine_res101_voc.yaml --dist-url tcp://192.168.16.38:23456 --ngpu 4 --world-size 6 --rank 4 --multiprocessing-distributed --workers 4
