all: lorra_wo_resnet, baseline

v_mmgnn:
	screen -S vgnn_try32 -m -d python tools/run.py --tasks vqa --datasets textvqa --model v_mmgnn \
		--config configs/vqa/textvqa/v_mmgnn.yml --seed 1234 -dev cuda:2

s_mmgnn:
	screen -S sgnn_try32 -m -d python tools/run.py --tasks vqa --datasets textvqa --model s_mmgnn \
		--config configs/vqa/textvqa/s_mmgnn.yml --seed 1234 -dev cuda:2

lorra_wo_resnet:
	screen -S wo_resnet1 -m -d python tools/run.py --tasks vqa --datasets textvqa --model lorra_wo_resnet \
		--config configs/vqa/textvqa/lorra_wo_resnet.yml --seed 1234 -dev cuda:1

baseline:
	screen -S trimmed0 -m -d python tools/run.py --tasks vqa --datasets textvqa --model lorra \
		--config configs/vqa/textvqa/lorra.yml --seed 1234 -dev cuda:0
