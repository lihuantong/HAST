CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node 1 main_direct.py --conf_path ./config/cifar10_resnet20.hocon