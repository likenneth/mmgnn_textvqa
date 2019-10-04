all: wo_resnet, baseline, sgm_mmgnn, v_mmgnn

v_mmgnn:
	screen -S v1_vgnn3 -m -d python tools/run.py --tasks vqa --datasets textvqa --model v_mmgnn \
		--config configs/vqa/textvqa/v_mmgnn.yml --seed 1234 -dev cuda:3

s_mmgnn:
	screen -S s_mmgnn1 -m -d python tools/run.py --tasks vqa --datasets textvqa --model s_mmgnn \
		--config configs/vqa/textvqa/s_mmgnn.yml --seed 1999 -dev cuda:1

wo_resnet:
	screen -S wo_resnet2 -m -d python tools/run.py --tasks vqa --datasets textvqa --model lorra_wo_resnet \
		--config configs/vqa/textvqa/lorra_wo_resnet.yml --seed 1234 -dev cuda:2

baseline:
	screen -S baseline0 -m -d python tools/run.py --tasks vqa --datasets textvqa --model lorra \
		--config configs/vqa/textvqa/lorra.yml --seed 1234 -dev cuda:0

sgm_mmgnn:
	screen -S sgm1 -m -d python tools/run.py --tasks vqa --datasets textvqa --model sgm_mmgnn \
		--config configs/vqa/textvqa/sgm_mmgnn.yml --seed 1234 -dev cuda:1
