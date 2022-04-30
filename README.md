<!--
 * @Date: 2022-03-06 10:53:36
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-04-30 15:06:47
-->
# PointAAE

In this project, we try to edit the latent representation of PointAAE and generate new reasonable point clouds from the operated latent representation. 

<img src="./img/pointaae.png" alt="pointaae" width="800"/>

## Setup

```bash
conda create -n pointaae python=3.6
conda activate pointaae

# Please check the PyTorch version: 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# All the other requirements are in: 
pip install -r requirements.txt

# -----------------------------------------------------------------------
# The following package is only compatible to CUDA 10.0. If your cuda's 
# version != 10.0, please do not install the following package. 
# You could compute the metric slowly with cpu, or you could chooes not to 
# comput MMD-CD and MMD-EMD. 
# The metrics are setted in `./settings/*.json:"metrics":["jsd", "mmd"]`.
# -----------------------------------------------------------------------
# Compile CUDA kernel for CD/EMD loss
root=`pwd`
cd metrics/pytorch_structural_losses/
make clean
make
cd $root
```

## Train

```bash
# Please check the settings, especially the cuda and gpu. 
python train_aae.py --config ./settings/init_exp.json

python train_aae.py --config ./settings/enc_exp.json
```

A visualization of reconstruction during training:

<img src="./img/init_res.png" alt="init_res" width="500"/>



## Eval

We didn't implement the results in the paper. It is expected to train the model for more than 2000 epochs as the authors did in the paper, but it crashed after 400 iterations. 

```bash
> python eval_aae.py --config ./settings/init_exp.json

2022-04-30 15:05:28,720: DEBUG    Minimum generation JSD at epoch 50:  0.596749
2022-04-30 15:05:28,721: DEBUG    Minimum reconstruction JSD at epoch 150:  0.054483
```

```bash
> python eval_aae.py --config ./settings/enc_exp.json

2022-04-30 14:51:20,269: DEBUG    Minimum generation JSD at epoch 150:  0.779633
2022-04-30 14:51:20,270: DEBUG    Minimum reconstruction JSD at epoch 285:  0.153872
```

## Edit

```bash
python edit_aae.py --config ./settings/init_exp.json --epoch 400
```

A visualization of editing (sum two embedded vectors): 

<img src="./img/edit_res.png" alt="edit_res" width="500"/>



## References:

- https://github.com/MaciejZamorski/3d-AAE

- https://github.com/AnTao97/dgcnn.pytorch

- https://github.com/stevenygd/PointFlow