<!--
 * @Date: 2022-03-06 10:53:36
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-04-24 16:03:30
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
```

A visualization of reconstruction during training:

<img src="./img/init_res.png" alt="init_res" width="500"/>

## Eval

```bash
python eval_aae.py --config ./settings/init_exp.json
```

After 300 epoch training: 

```bash
2022-04-06 12:03:25,281: DEBUG    Evaluating JensenShannon divergences on validation set on all saved epochs.
2022-04-06 12:03:25,282: DEBUG    Testing epochs: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295]
2022-04-06 12:03:25,857: DEBUG    Device variable: cuda
2022-04-06 12:03:25,857: DEBUG    Current CUDA device: 1
2022-04-06 12:03:26,855: DEBUG    Selected all classes. Loaded 2870 samples.
2022-04-06 12:06:07,337: DEBUG    Epoch: 295 JSD:  0.658785 Time: 0:02:37.607074
2022-04-06 12:08:41,789: DEBUG    Epoch: 290 JSD:  0.644904 Time: 0:02:34.170156
2022-04-06 12:11:10,268: DEBUG    Epoch: 285 JSD:  0.652403 Time: 0:02:28.201789
2022-04-06 12:13:45,225: DEBUG    Epoch: 280 JSD:  0.649380 Time: 0:02:34.684850
2022-04-06 12:16:08,869: DEBUG    Epoch: 275 JSD:  0.651107 Time: 0:02:23.347013
2022-04-06 12:18:22,778: DEBUG    Epoch: 270 JSD:  0.669900 Time: 0:02:13.646959
2022-04-06 12:21:02,326: DEBUG    Epoch: 265 JSD:  0.667532 Time: 0:02:39.274505
^C2022-04-06 12:21:25,830: DEBUG    Interrupted during epoch: 260
2022-04-06 12:21:25,850: DEBUG    Minimum JSD at epoch 290:  0.644904
```

We are expected to train it for more than 2000 epochs as the author did in the paper. New results are coming soon.  

## Edit

```bash
python edit_aae.py --config ./settings/init_exp.json --epoch 400
```



## References:

https://github.com/MaciejZamorski/3d-AAE