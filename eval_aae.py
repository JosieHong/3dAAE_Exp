'''
Date: 2022-04-06 11:57:32
LastEditors: yuhhong
LastEditTime: 2022-05-02 20:39:51
'''
import os
import argparse
import json
import logging
import random
import re
from datetime import datetime
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.shapenet import ShapeNetDataset
from metrics.jsd import jsd_between_point_cloud_sets
from metrics.mmd import mmd_between_point_cloud_sets
from utils.util import cuda_setup, setup_logging


def _get_epochs_by_regex(path, regex):
    reg = re.compile(regex)
    return {int(w[:5]) for w in os.listdir(path) if reg.match(w)}


def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = os.path.join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(os.path.join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    setup_logging(os.path.join(train_results_path, 'results'))
    log = logging.getLogger(__name__)

    log.debug('Evaluation on validation set on all saved epochs.')

    weights_path = os.path.join(train_results_path, 'weights')

    # Find all epochs that have saved model weights
    e_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})_E\.pth')
    g_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})_G\.pth')
    epochs = sorted(e_epochs.intersection(g_epochs))
    log.debug(f'Testing epochs: {epochs}')

    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    # ########################################
    # Dataset
    # ########################################
    dataset_name = train_config['dataset'].lower()
    if dataset_name == 'shapenet':
        dataset = ShapeNetDataset(root_dir=train_config['data_dir'],
                                  classes=train_config['classes'], split='valid')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` '
                         f'Got: `{dataset_name}`')
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))
    log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')

    if 'distribution' in train_config:
        distribution = train_config['distribution']
    elif 'distribution' in eval_config:
        distribution = eval_config['distribution']
    else:
        log.warning('No distribution type specified. Assumed normal = N(0, 0.2)')
        distribution = 'normal'

    # ########################################
    # Models
    # ########################################
    # arch = import_module(f"model.architectures.{train_config['arch']}")
    arch = import_module(f"models.{train_config['arch']}")
    E = arch.Encoder(train_config).to(device)
    G = arch.Generator(train_config).to(device)

    E.eval()
    G.eval()

    # num_samples = len(dataset.point_clouds_names_valid)
    num_samples = eval_config['batch_size']
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    # noise = torch.FloatTensor(3 * num_samples, train_config['z_size'], 1)
    # J0sie: Why?
    noise = torch.FloatTensor(num_samples, train_config['z_size'], 1)
    noise = noise.to(device)

    X, _ = next(iter(data_loader))
    X = X.to(device)

    assert len(eval_config['metrics']) != 0
    if 'jsd' in eval_config['metrics'] and 'mmd' in eval_config['metrics']:
        results = {'epoch': [], 'gen_jsd': [], 'rec_jsd': [], 'gen_mmd-cd': [], 
                    'rec_mmd-cd': [], 'gen_mmd-emd': [], 'rec_mmd-emd': []}
    elif 'jsd' in eval_config['metrics']:
        results = {'epoch': [], 'gen_jsd': [], 'rec_jsd': []}
    elif 'mmd' in eval_config['metrics']:
        results = {'epoch': [], 'gen_mmd-cd': [], 'rec_mmd-cd': [], 
                    'gen_mmd-emd': [], 'rec_mmd-emd': []}

    for epoch in reversed(epochs): 
        try:
            E.load_state_dict(torch.load(
                os.path.join(weights_path, f'{epoch:05}_E.pth')))
            G.load_state_dict(torch.load(
                os.path.join(weights_path, f'{epoch:05}_G.pth')))

            start_clock = datetime.now()

            # ########################################
            # Reconstruction
            # ########################################
            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            X.transpose_(X.dim() - 2, X.dim() - 1)
            # inference
            with torch.no_grad(): 
                codes, _, _ = E(X) # codes.size(): torch.Size([64, 2048])
                X_r = G(codes)
            # post-process
            X_r.transpose_(1, 2)
            X.transpose_(1, 2)

            # metrics
            if 'jsd' in eval_config['metrics']:
                try:
                    jsd_rec = jsd_between_point_cloud_sets(X, X_r, voxels=28)
                except ValueError: 
                    continue

            if 'mmd' in eval_config['metrics']: 
                try:
                    cd_rec, emd_rec = mmd_between_point_cloud_sets(X, X_r, 
                                    batch_size=eval_config['batch_size'], reduced=True)
                except ValueError: 
                    continue
            
            # ########################################
            # Generateion
            # ########################################
            # We average JSD computation from 3 independet trials.
            js_results = []
            cd_results = []
            emd_results = []
            for _ in range(3): 
                if distribution == 'normal':
                    noise.normal_(0, 0.2)
                elif distribution == 'beta':
                    noise_np = np.random.beta(train_config['z_beta_a'],
                                              train_config['z_beta_b'],
                                              noise.shape)
                    noise = torch.tensor(noise_np).float().round().to(device)

                with torch.no_grad():
                    X_g = G(noise)
                # post-process
                X_g.transpose_(1, 2)

                if 'jsd' in eval_config['metrics']:
                    try:
                        jsd = jsd_between_point_cloud_sets(X, X_g, voxels=28)
                        js_results.append(jsd)
                    except ValueError: 
                        # log.debug(f'NaN result in epoch: {epoch}')
                        continue

                if 'mmd' in eval_config['metrics']: 
                    try:
                        cd, emd = mmd_between_point_cloud_sets(X, X_g, 
                                        batch_size=eval_config['batch_size'], reduced=True)
                        cd_results.append(cd.item())
                        emd_results.append(emd.item())
                    except ValueError: 
                        # log.debug(f'NaN result in epoch: {epoch}')
                        continue
                    
            results['epoch'].append(epoch)
            if 'jsd' in eval_config['metrics']:
                try:
                    js_result = np.mean(js_results)
                except ValueError: 
                    js_result = np.nan

                log.debug(f'Epoch: {epoch} '
                            f'Gen JSD: {js_result: .6f} '
                            f'Rec JSD: {jsd_rec: .6f} '
                            f'Time: {datetime.now() - start_clock}')
                results['gen_jsd'].append(js_result)
                results['rec_jsd'].append(jsd_rec.item())

            if 'mmd' in eval_config['metrics']: 
                try:
                    cd_result = np.mean(cd_results)
                    emd_result = np.mean(emd_results)
                except ValueError: 
                    cd_result = np.nan
                    emd_result = np.nan

                log.debug(f'Epoch: {epoch} '
                            f'Gen MMD-CD: {cd_result: .6f} Gen MMD-EMD: {emd_result: .6f} '
                            f'Rec MMD-CD: {cd_rec: .6f} Rec MMD-EMD: {emd_rec: .6f} '
                            f'Time: {datetime.now() - start_clock}')
                results['gen_mmd-cd'].append(cd_result)
                results['gen_mmd-emd'].append(emd_result)
                results['rec_mmd-cd'].append(cd_rec.item())
                results['rec_mmd-emd'].append(emd_rec.item())

        except KeyboardInterrupt:
            log.debug(f'Interrupted during epoch: {epoch}')
            break

    # results = pd.DataFrame.from_dict(results, orient='index', columns=['jsd'])
    results = pd.DataFrame.from_dict(results).set_index('epoch')
    if 'jsd' in eval_config['metrics']:
        log.debug(f"Minimum generation JSD at epoch {results.idxmin()['gen_jsd']}: "
                f"{results.min()['gen_jsd']: .6f} ")
        log.debug(f"Minimum reconstruction JSD at epoch {results.idxmin()['rec_jsd']}: "
                f"{results.min()['rec_jsd']: .6f} ")
    if 'mmd' in eval_config['metrics']:
        log.debug(f"Minimum generation MMD-CD at epoch {results.idxmin()['gen_mmd-cd']}: "
                f"{results.min()['gen_mmd-cd']: .6f} "
                f"Minimum generation MMD-EMD at epoch {results.idxmin()['gen_mmd-emd']}: "
                f"{results.min()['gen_mmd-emd']: .6f} "
                f"Minimum reconstruction MMD-CD at epoch {results.idxmin()['rec_mmd-cd']}: "
                f"{results.min()['rec_mmd-cd']: .6f} "
                f"Minimum reconstruction MMD-EMD at epoch {results.idxmin()['rec_mmd-emd']}: "
                f"{results.min()['rec_mmd-emd']: .6f} ")
    # save the results
    results.to_csv(os.path.join(train_results_path, "restuls_{}.csv".format("_".join(classes_selected.split(',')))))
    log.debug(f'Save the final results.')

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)