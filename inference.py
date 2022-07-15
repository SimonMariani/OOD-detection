import numpy as np
import torch

import os
import yaml
from pathlib import Path
from distutils.dir_util import copy_tree
from types import SimpleNamespace   
import time
import pickle

from utils import check_paths

from setup import get_experiment, get_data_id, get_data_ood, get_model, get_model_unc


def infer(model_unc, dataloaders, names, device='cuda:0', num_batches=None, print_every=None, save_dir=None, exp_name='test'):
    r"""
    Performs inference on an uncertainty model and returns the logits, targets, uncertainties and optional model logits.
    """
    
    model_unc.to(device)
    model_unc.eval()

    if not model_unc.needs_grad:
        torch.set_grad_enabled(False)
    
    all_data = []
    start_time = time.time()
    previous_time = time.time()
    for dataloader, name in zip(dataloaders, names):
        
        print(f'inferring {name}')

        all_logits = []
        all_targets = []
        all_unc = []

        for j, (imgs, targets) in enumerate(dataloader):

            imgs, targets = imgs.to(device), targets.to(device)
           
            targets = torch.argmax(targets, dim=1) if len(targets.shape) > 1 else targets
            logits = model_unc(imgs)
            uncertainties = model_unc.uncertainty(imgs)
            
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_unc.append(uncertainties.detach().cpu())
            
            if num_batches is not None:
                if j >= num_batches:
                    break

            if print_every is not None:
                if j % print_every == 0:
                    print("batch: ", j)

        all_logits, all_targets, all_unc = torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0), torch.cat(all_unc, dim=0)
        
        sub_time = time.time() - previous_time
        previous_time = time.time()

        print(f'finished inference, time: {str(np.round(sub_time, 3)):6s}s, {str(np.round(sub_time / 60, 3)):6s}m,',
              f'{str(np.round(sub_time / 3600, 3)):6s}h')

        if save_dir is not None:
            save_path = save_dir + exp_name +  f'_{name}.pickle'
            with open(save_path, 'wb') as handle:
                pickle.dump((all_logits, all_targets, all_unc), handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        all_data.append( (all_logits, all_targets, all_unc) )

    total_time = time.time() - start_time
    print(f'\nfinished inference, time: {str(np.round(total_time, 3)):6s}s, {str(np.round(total_time / 60, 3)):6s}m,',
              f'{str(np.round(total_time / 3600, 3)):6s}h')

    torch.set_grad_enabled(True)
    
    return all_data

def run(data_dir, save_dir, save_dir_par, config, seed, exp_name, device):

    print(f'\nSTARTED EXPERIMENT: {exp_name}')
    config_inf = SimpleNamespace(**config.inference)
    config = SimpleNamespace(**config.main_run)

    print("\nCREATING THE SAVE DIRECTORY")
    save_dir_inf = save_dir + 'inference/'
    Path(save_dir_inf).mkdir(parents=True, exist_ok=True)
    print(f'saving data to {save_dir_inf}')

    print("\nLOADING THE ID DATA")
    trainloader, testloader, num_classes = get_data_id(config, data_dir, seed, device, train=False)

    print("\nLOADING THE OOD DATASETS")
    testloaders_ood = get_data_ood(config, config_inf.dataset_ood_test, data_dir, seed, device)

    print("\nLOADING THE MODEL AND UNCERTAINTY MODEL")
    config.pretrained_path_unc = save_dir_par + 'state_dicts/' + exp_name + '_uncertain_post.pth' if config.pretrained_path_unc is None else config.pretrained_path_unc
    model = get_model(config, num_classes)
    model_unc = get_model_unc(model, config, num_classes, trainloader=trainloader)

    print("\nINFERRING DATA LOADERS")
    loaders = [testloader] + testloaders_ood if config_inf.infer_id else testloaders_ood
    names = [config.dataset] + config_inf.dataset_ood_test if config_inf.infer_id else config_inf.dataset_ood_test

    loaders = loaders + [trainloader] if config_inf.infer_train else loaders
    names = names + [config.dataset + '_train'] if config_inf.infer_train else names

    infer(model_unc, loaders, names, device=device, save_dir=save_dir_inf, exp_name=exp_name)

    print('\nCOPYING DATA TO RESULTS DIRECTORY')
    copy_tree(save_dir, save_dir_par)
    print(f'copied data to directory {save_dir_par}')

    print(f'\nFINISHED EXPERIMENT: {exp_name}')


if __name__ == '__main__':

    with open('experiment.yaml') as f:
        experiment = yaml.load(f, yaml.SafeLoader)
        experiment = SimpleNamespace(**experiment)

    if isinstance(experiment.config_files, list):
        config_files = experiment.config_files
    else:
        if experiment.config_files.endswith('.yaml'):
            config_files = [experiment.config_files]
        else:
            config_files = [file for file in os.listdir(experiment.config_files)]
    
    print('\nINFERRING ON FILES')
    print(config_files)
    check_paths(experiment.configs_dir, config_files)

    for config in config_files:

        experiment.config_file = config
        data_dir, save_dir, save_dir_par, config, seed, exp_name, device = get_experiment(experiment)

        run(data_dir, save_dir, save_dir_par, config, seed, exp_name, device)
