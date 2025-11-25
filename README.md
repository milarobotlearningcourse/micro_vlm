[![Open micro-vlm In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/milarobotlearningcourse/micro_vlm/blob/main/micro_vlm.ipynb)

# Micro Vision Language Model: 

Minimialist reimplimentation of a decoder only Vision Language Model (VLM).

## Install

conda create -n micro-vlm python=3.10
conda activate micro-vlm
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Dataset

https://huggingface.co/datasets/merve/vqav2-small
Has around 21k datapoints for VQA.

The code reduces the image size down to 64x64x3 so it can fit on a small computer.

## Running the code

Basic example to train the GRP over the bridge dataset 

```
python micro_vlm.py
```

Launch multiple jobs on a slurm cluster to evalute different model architectures, etc.
```
python mini-grp.py --multirun gradient_accumulation_steps=1,2,4 hydra/launcher=submitit_slurm
```


### License

MIT
