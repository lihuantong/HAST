for g in 1 2 3 4
do
python generate_data.py 		\
		--model=resnet20_cifar10 			\
		--batch_size=256 		\
		--test_batch_size=512 \
		--group=$g \
		--beta=10 \
		--gamma=2 \
		--save_path_head=../data/cifar10
done
