#!/bin/bash
export WANDB_API_KEY='2afe2846a028d2bc651361a4fbdf85cfbd9d9eab'
wandb login

servername="A600"
# 初始化另一个字符串的变量
if [ "$servername" == "rc" ]; then
    server=/home/tw9146/tw
    cd /home/tw9146/tw
else
    server=/spl_data/tw9146
    cd /spl_data/tw9146
fi

# 输出结果以验证
echo "$server"
#tags="LRDecay WarmUp single DoF7 geometry colorjitter debug"
tags="debug"
python openvla-main/attack-oneDOF-PositionORrotation/attack_white_patch.py --lr 2e-3 --maskidx 0,1,2 --server $server --device 0 --iter 100000 --accumulate 1 --bs 6 --warmup 200 --tags $tags --filterGripTrainTo1 false --geometry true --colorjitter true


