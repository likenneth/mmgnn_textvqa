## Multi-Modal GNN for TextVQA

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.0.1-%237732a8)

+ This project provides codes to reproduce the results of 
[Multi-Modal Graph Neural Network for Joint Reasoning on Vision and Scene Text](https://www.google.com/) on TextVQA dataset.  
+ We are grateful to [Pythia](https://github.com/facebookresearch/pythia "Pythia's Github repo"), an 
excellent VQA codebase provided by Facebook, on which our codes are developed.
+ We achieved 32.46% accuracy (ensemble) on test set of TextVQA

### Requirements

- Pytorch 1.0.1 post.
- We have performed experiments on Maxwell Titan X GPU. We assume 12GB of GPU memory.
- See [`requirements.txt`](requirements.txt) for the required python packages and run to install them.

Let's begin from cloning this repository
```
$ git clone https://github.com/ricolike/mmgnn-textvqa.git
$ cd mmgnn-textvqa
$ pip install -r requirements.txt
```

### Data Setup

1. **cached data:** To boost data loading speed under limited memory size (64G) and to speed
up calculation, we cached intermediate dataloader results in storage. Download 
[data](https://drive.google.com/drive/folders/1Y8E-afg9aRHn6VblSWGNd0hvQGEW9ILS?usp=sharing) 
*(around 54G, and around 120G unzipped)*, and modify 
line 11 (`fast_dir`) in [config](pythia/common/defaults/configs/tasks/vqa/textvqa.yml)
to the absolute path where you save them
2. **other files:** Download other needed files (vocabulary, OCRs, some parameters of 
backbone) [here](https://drive.google.com/file/d/1ieIx4MB49DBm1ycY203f15kvcrX4IoLt/view?usp=sharing) 
*(less than 1G)*, and make a soft link named `data` under repo root towards where you saved them

### Training
+ Create a new model folder under `ensemble`, say `foo`, and then copy [our config](configs/vqa/textvqa/s_mmgnn.yml) 
into it  
```
$ mkdir ensemble/foo
$ cp ./configs/vqa/textvqa/s_mmgnn.yml ./ensemble/foo
```
+ Start training, and parameters will be saved in `ensemble/foo`
```
$ python tools/run.py --tasks vqa --datasets textvqa --model s_mmgnn --config ensemble/foo/s_mmgnn.yml -dev cuda:0 --run_type train`
```
+ First-run of this repo will automatically download glove in `pythia/.vector_cache`, 
let's be patient. If we made it, we will find a `s_mmgnnbar_final.pth` in the model folder `ensemble/foo`

### Inference

+ If training is undesired, 
a [trained](https://drive.google.com/file/d/1P1k3sNAQnV7dUovypt1zKwCTNgCEDHua/view?usp=sharing) model is provided
on which we can directly do inference
+ Start inference by
```
$ python tools/run.py --tasks vqa --datasets textvqa --model s_mmgnn --config ensemble/bar/s_mmgnn.yml --resume_file <path_to_pth> -dev cuda:0 --run_type all_in_one
```
, and if we made it, we will find three new files generated under the model folder, two 
ends with `_evailai.p` are ready to be submitted to 
[evalai](https://evalai.cloudcv.org/web/challenges/challenge-page/244/leaderboard/809) to
check the results

### Bibtex
```
@article{...,
  title={Multi-Modal Graph Neural Network for Joint Reasoning on Vision and Scene Text},
  author={...},
  journal={...},
  year={2020}
}
```

### An attention visualization

![](pics/high_res.png)  
**_Question: "What is the name of the bread sold at the place?"_**  
**_Answer: "Panera"_**  
(where white box is the answer predicted, green boxes are OCRs **_Panera_** attends to, and 
red boxes are visual ROIs **_Panera_** attends to; box weight indicating attention strength)
