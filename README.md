## Requirements

### Installation

```bash
we use /torch >=2.1.0 / 24G  RTX3090 for training and evaluation.
```
install mamba-ssm:
```bash
pip install mamba-ssm==1.2.2
```
or install it through a wheel file, Here is the link to the wheel file:
[mamba-ssm](https://github.com/state-spaces/mamba/releases/tag/v1.2.2)

### Prepare Datasets

Download the HDR datasets [Kalantari](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/).

Then unzip them and rename them under the directory like.

```
data
├── Training
│   └── 
│   └── 
│   └── 
├── Test
│   └── 
│   └── 
│   └── 
```

### Train
```bash
python train.py
```



## Evaluation

```bash
python test.py
```
