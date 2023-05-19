CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
-a resnet50 -b 64 --test-batch-size 256 --iters 200 --lr 3.5e-4 --epoch 70 --seed 2 \
--dataset_src1 msmt17v1 --dataset_src2 cuhk03 --dataset_src3 market1501 -d dukemtmc --updateStyle \
--logs-dir logs/IL \
--data-dir /data3/guowei/reidData/