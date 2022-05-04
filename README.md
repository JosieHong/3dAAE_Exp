<!--
 * @Date: 2022-03-06 10:53:36
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-05-04 15:15:32
-->
# B659 Experiments on 3dAAE

This is the final project of CSCI-B659 Computer Vision, Indiana University. In this project, we try to edit the latent representation of [3dAAE](https://arxiv.org/abs/1811.07605) and generate new reasonable point clouds from the operated latent representation. Based on the original codes of 3dAAE, we implement the following things: 

- [x] Schedulers in training, `./train_aae.py:EG_scheduler` and `./train_aae.py:D_scheduler`;
- [x] MMD-CD and MMD-EMD in evaluation metrics, `./metrics/mmd.py`;
- [x] Editing the vectors and generate point couds, `./edit_aae.py`;
- [x] Different encoders, `./models/dgcnn_aae.py`; However, it did not perform good so far, so we did not show its in the final report. 

The workflow of our experiments:
 
<img src="./img/pointaae.png" alt="pointaae" width="70%"/>

If you feel this experiment is inspirable, please cite the original paper of `3dAAE`: 

```
@article{zamorski2018adversarial,
  title={Adversarial Autoencoders for Compact Representations of 3D Point Clouds},
  author={Zamorski, Maciej and Zi{\k{e}}ba, Maciej and Klukowski, Piotr and Nowak, Rafa{\l} and Kurach, Karol and Stokowiec, Wojciech and Trzci{\'n}ski, Tomasz},
  journal={arXiv preprint arXiv:1811.07605},
  year={2018}
}
```



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
# version != 10.0, please DO NOT install the following package. 
# You could compute the metric slowly without cuda, or you could chooes not to 
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

python train_aae.py --config ./settings/dgcnn_enc_exp.json
```

A visualization of reconstruction during training:

<img src="./img/init_res.png" alt="init_res" width="50%"/>



## Eval

We implement JSD, MMD-CD and MMD-EMD for evaluation. Please chooes the metrics in settings, for instance `"metrics": ["jsd", "mmd"]`.  

```bash
python eval_aae.py --config ./settings/init_exp.json

python eval_aae.py --config ./settings/dgcnn_enc_exp.json
```



## Edit

```bash
python edit_aae.py --config ./settings/init_exp.json --epoch 400
```

A visualization of editing (sum two embedded vectors): 

<img src="./img/edit_res.png" alt="edit_res" width="500"/>



## References:

Our experiments are mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- https://github.com/MaciejZamorski/3d-AAE

- https://github.com/AnTao97/dgcnn.pytorch

- https://github.com/stevenygd/PointFlow