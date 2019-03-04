#!/bin/bash
IP=`/sbin/ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"`
TCPADDR=tcp://$IP:23456
WORLDSIZE=6
MAINEXEC='envpy3/bin/python train_dist.py --save_folder ./dist --cfg configs/refine_res101_voc.yaml --dist-url '$TCPADDR'  --ngpu 4 --world-size '$WORLDSIZE' --rank 0 --multiprocessing-distributed --workers 4'
echo $MAINEXEC >'train.sh'

INSTALLNCCL='wget http://pbsdv028w.bkt.clouddn.com/softwares/nvidia/nccl-repo-ubuntu1604-2.4.2-ga-cuda9.0_1-1_amd64.deb\ndpkg -i nccl-repo-ubuntu1604-2.4.2-ga-cuda9.0_1-1_amd64.deb' 

for((integer = WORLDSIZE-1; integer >= 1; integer--))
do
    MAINEXEC='cd /workspace/mnt/group/test/zhaozhijian/distributed/RefineDetPytorch/\nenvpy3/bin/python train_dist.py --save_folder ./dist --cfg configs/refine_res101_voc.yaml --dist-url '$TCPADDR'  --ngpu 4 --world-size '$WORLDSIZE' --rank '$integer' --multiprocessing-distributed --workers 4 > log_'$integer'.txt 2>&1 &'
    echo -e $INSTALLNCCL>'train'$integer'.sh'
    echo -e $MAINEXEC>>'train'$integer'.sh'
done

