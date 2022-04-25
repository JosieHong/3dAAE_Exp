'''
Date: 2022-04-24 15:04:36
LastEditors: yuhhong
LastEditTime: 2022-04-25 15:39:49
'''
import os
import logging
import json
import argparse
import random
from importlib import import_module
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from datasets.shapenet import ShapeNetDataset
from utils.util import cuda_setup, setup_logging
from utils.pcutil import plot_3d_point_cloud, plot_codes



logging.getLogger('matplotlib.font_manager').disabled = True

def main(eval_config, epoch, operation): 
    '''Load the datasets and models with checkpoints of a specific epoch
    '''

    # Load hyperparameters as they were during training
    train_results_path = os.path.join(eval_config['results_root'], eval_config['arch'],
                                        eval_config['experiment_name'])
    with open(os.path.join(train_results_path, 'config.json')) as f: 
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    setup_logging(os.path.join(train_results_path, 'editing'))
    log = logging.getLogger(__name__)
    log.debug('Editing the codes in latent space and generate new points.')

    weights_path = os.path.join(train_results_path, 'weights')

    # Use a specific epoch for editing
    log.debug(f'Editing epoch: {epoch}')

    #
    # Device
    #
    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = train_config['dataset'].lower()
    if dataset_name == 'shapenet':
        dataset = ShapeNetDataset(root_dir=train_config['data_dir'],
                                  classes=train_config['classes'], split='valid')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet`. Got: `{dataset_name}`')
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))
    log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')
    # Load the dataset
    data_loader = DataLoader(dataset, batch_size=eval_config['batch_size'],
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)
    
    #
    # Models
    #
    arch = import_module(f"models.{train_config['arch']}")
    E = arch.Encoder(train_config).to(device)
    G = arch.Generator(train_config).to(device)
    E.eval()
    G.eval()
    # Load the checkpoints
    E.load_state_dict(torch.load(
        os.path.join(weights_path, f'{epoch:0>5}_E.pth')))
    G.load_state_dict(torch.load(
        os.path.join(weights_path, f'{epoch:0>5}_G.pth')))

    #
    # Editing
    # 
    for i, point_data in enumerate(data_loader): 
        # 1. Get the encoding results in latent space 
        X, _ = point_data
        X = X.to(device)
        
        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)
        
        with torch.no_grad(): 
            codes, _, _ = E(X) # codes.size(): torch.Size([64, 2048])

        # 2. Edit all the codes (add next)
        new_codes = torch.zeros_like(codes)
        for k in range(codes.size()[0]): 
            if operation == '+':
                new_codes[k] = 0.5 * codes[k] + 0.5 * codes[(k+1) % codes.size()[0]]
            elif operation == '-':
                new_codes[k] = codes[k] - codes[(k+1) % codes.size()[0]]

        # 3. Generate the points clouds 
        with torch.no_grad():
            X_g = G(codes)
            new_X_g = G(new_codes)
        
        # 4. Show edited codes and generated point clouds
        for k in range(eval_config['edit_samples']): 
            idx = i * eval_config['batch_size'] + k

            # real point cloud
            fig = plot_3d_point_cloud(X[k][0].cpu(), X[k][1].cpu(), X[k][2].cpu(),
                                      in_u_sphere=True, show=False)
            fig.savefig(
                os.path.join(train_results_path, 'editing', f'{epoch:0>5}_{idx}_real.png'))
            plt.close(fig)

            # real codes
            fig = plot_codes(codes[k].cpu(), show=False, show_axis=False, 
                            figsize=(10, 2.5))
            fig.savefig(
                os.path.join(train_results_path, 'editing', f'{epoch:0>5}_{idx}_real_codes.png'))
            plt.close(fig)

            # reconstructed point cloud
            fig = plot_3d_point_cloud(X_g[k][0].cpu(), X_g[k][1].cpu(), X_g[k][2].cpu(),
                                      in_u_sphere=True, show=False)
            fig.savefig(
                os.path.join(train_results_path, 'editing', f'{epoch:0>5}_{idx}_reconstructed.png'))
            plt.close(fig)

            # generated point cloud
            fig = plot_3d_point_cloud(new_X_g[k][0].cpu(), new_X_g[k][1].cpu(), new_X_g[k][2].cpu(),
                                      in_u_sphere=True, show=False)
            fig.savefig(
                os.path.join(train_results_path, 'editing', f'{epoch:0>5}_{idx}_gen.png'))
            plt.close(fig)

            # edited codes
            fig = plot_codes(new_codes[k].cpu(), show=False, show_axis=False, 
                            figsize=(10, 2.5))
            fig.savefig(
                os.path.join(train_results_path, 'editing', f'{epoch:0>5}_{idx}_gen_codes.png'))
            plt.close(fig)



if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    parser.add_argument('-e', '--epoch', default=None, type=str,
                        help='Please choes an epoch for editing')
    parser.add_argument('--operation', type=str, default='+', choices=['+', '-'], 
                        help='Operation used to edit the embedded vector')
    args = parser.parse_args()

    eval_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            eval_config = json.load(f)
    assert eval_config is not None

    main(eval_config, args.epoch, args.operation)
    
    
