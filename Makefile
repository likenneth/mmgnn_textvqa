all: wo_resnet, baseline, sgm_mmgnn, v_mmgnn

lorra:
	screen -S s_mmgnn0 -m -d python tools/run.py --tasks vqa --datasets textvqa --model s_mmgnn \
		--config configs/vqa/textvqa/s_mmgnn.yml --seed 1234 -dev cuda:0 --run_type train

s_mmgnn:
	screen -S s_mmgnn2 -m -d python tools/run.py --tasks vqa --datasets textvqa --model s_mmgnn \
		--config /home/like/Workplace/textvqa/ensemble/ss_cooling/s_mmgnn.yml \
		 --seed 1234 -dev cuda:2 --run_type train

wo_resnet:
	screen -S wo_resnet2 -m -d python tools/run.py --tasks vqa --datasets textvqa --model lorra_wo_resnet \
		--config configs/vqa/textvqa/lorra_wo_resnet.yml --seed 1234 -dev cuda:2

baseline:
	screen -S baseline0 -m -d python tools/run.py --tasks vqa --datasets textvqa --model lorra \
		--config configs/vqa/textvqa/lorra.yml --seed 1234 -dev cuda:0

sgm_mmgnn:
	screen -S sgm1 -m -d python tools/run.py --tasks vqa --datasets textvqa --model sgm_mmgnn \
		--config configs/vqa/textvqa/sgm_mmgnn.yml --seed 1234 -dev cuda:1
