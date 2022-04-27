<!--
 * @Date: 2022-03-06 10:53:36
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-04-27 13:21:54
-->
# PointAAE

In this project, we try to edit the latent representation of PointAAE and generate new reasonable point clouds from the operated latent representation. 

<img src="./img/pointaae.png" alt="pointaae" width="500"/>

## Setup

```bash
conda create -n pointaae python=3.6
conda activate pointaae

# Please check the PyTorch version: 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# All the other requirements are in: 
pip install -r requirements.txt
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

```bash
python eval_aae.py --config ./settings/init_exp.json

python eval_aae.py --config ./settings/enc_exp.json
```

We are expected to train it for more than 2000 epochs as the author did in the paper, but it crashed after 400 iterations. Let's find out what happened. 

```bash
2022-04-25 12:06:05,489: DEBUG    Evaluating JensenShannon divergences on validation set on all saved epochs.
2022-04-25 12:06:05,490: DEBUG    Testing epochs: [50, 100, 150, 200, 250, 300, 350, 400, 450]
2022-04-25 12:06:06,159: DEBUG    Device variable: cuda
2022-04-25 12:06:06,159: DEBUG    Current CUDA device: 0
2022-04-25 12:06:07,179: DEBUG    Selected all classes. Loaded 2870 samples.
2022-04-25 12:08:45,067: DEBUG    Epoch: 450 JSD:  0.653789 Time: 0:02:33.634625
2022-04-25 12:11:20,719: DEBUG    Epoch: 400 JSD:  0.655928 Time: 0:02:35.586871
2022-04-25 12:13:54,555: DEBUG    Epoch: 350 JSD:  0.657119 Time: 0:02:33.597101
2022-04-25 12:16:34,994: DEBUG    Epoch: 300 JSD:  0.663368 Time: 0:02:40.202922
2022-04-25 12:19:15,950: DEBUG    Epoch: 250 JSD:  0.663630 Time: 0:02:40.750014
2022-04-25 12:21:59,826: DEBUG    Epoch: 200 JSD:  0.649842 Time: 0:02:43.644191
2022-04-25 12:24:40,552: DEBUG    Epoch: 150 JSD:  0.656921 Time: 0:02:40.529231
2022-04-25 12:27:14,297: DEBUG    Epoch: 100 JSD:  0.666355 Time: 0:02:33.505640
2022-04-25 12:29:49,034: DEBUG    Epoch: 50 JSD:  0.663614 Time: 0:02:34.500206
2022-04-25 12:29:49,043: DEBUG    Minimum JSD at epoch 200:  0.649842
```

```
2022-04-27 12:29:58,510: DEBUG    Evaluating JensenShannon divergences on validation set on all saved epochs.
2022-04-27 12:29:58,511: DEBUG    Testing epochs: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
2022-04-27 12:29:59,091: DEBUG    Device variable: cuda
2022-04-27 12:29:59,091: DEBUG    Current CUDA device: 0
2022-04-27 12:30:00,089: DEBUG    Selected all classes. Loaded 2870 samples.
2022-04-27 12:32:29,207: DEBUG    Epoch: 100 JSD:  0.683622 Time: 0:02:26.354480
2022-04-27 12:34:52,174: DEBUG    Epoch: 95 JSD:  0.696193 Time: 0:02:22.921197
2022-04-27 12:37:15,504: DEBUG    Epoch: 90 JSD:  0.692594 Time: 0:02:23.289040
2022-04-27 12:39:38,341: DEBUG    Epoch: 85 JSD:  0.692290 Time: 0:02:22.789913
2022-04-27 12:42:06,574: DEBUG    Epoch: 80 JSD:  0.687191 Time: 0:02:28.188051
2022-04-27 12:44:33,593: DEBUG    Epoch: 75 JSD:  0.689317 Time: 0:02:26.984130
2022-04-27 12:47:01,550: DEBUG    Epoch: 70 JSD:  0.680599 Time: 0:02:27.919062
2022-04-27 12:49:31,066: DEBUG    Epoch: 65 JSD:  0.676769 Time: 0:02:29.479045
2022-04-27 12:51:59,317: DEBUG    Epoch: 60 JSD:  0.686496 Time: 0:02:28.205194
2022-04-27 12:54:28,777: DEBUG    Epoch: 55 JSD:  0.675430 Time: 0:02:29.395281
2022-04-27 12:56:59,745: DEBUG    Epoch: 50 JSD:  0.681967 Time: 0:02:30.919885
2022-04-27 12:59:30,673: DEBUG    Epoch: 45 JSD:  0.693716 Time: 0:02:30.888530
2022-04-27 13:01:58,861: DEBUG    Epoch: 40 JSD:  0.703029 Time: 0:02:28.150603
2022-04-27 13:04:35,170: DEBUG    Epoch: 35 JSD:  0.713698 Time: 0:02:36.269071
2022-04-27 13:07:14,663: DEBUG    Epoch: 30 JSD:  0.718611 Time: 0:02:39.449553
2022-04-27 13:09:55,718: DEBUG    Epoch: 25 JSD:  0.734767 Time: 0:02:41.003776
2022-04-27 13:12:37,457: DEBUG    Epoch: 20 JSD:  0.731144 Time: 0:02:41.692943
2022-04-27 13:15:15,644: DEBUG    Epoch: 15 JSD:  0.748060 Time: 0:02:38.145803
2022-04-27 13:17:24,417: DEBUG    Epoch: 10 JSD:  0.756682 Time: 0:02:08.736856
2022-04-27 13:19:42,992: DEBUG    Epoch: 5 JSD:  0.777524 Time: 0:02:18.537891
2022-04-27 13:19:42,994: DEBUG    Minimum JSD at epoch 55:  0.675430
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