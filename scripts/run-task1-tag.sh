#!/bin/bash
arch=$1
pair=$2
highreslang=$(echo $pair|sed -e 's/--/:/g'|cut -d: -f1)
lowreslang=$(echo $pair|sed -e 's/--/:/g'|cut -d: -f2)
python src/train.py \
    --dataset sigmorphon19task1 \
    --train sample/$pair/$highreslang-train-high sample/$pair/$lowreslang-train-low  \
    --dev sample/$pair/$lowreslang-dev \
    --model model/sigmorphon19/task1/tag-$arch/$pair --seed 0 \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 \
    --arch $arch --estop 1e-8 --epochs 50 --bs 20
