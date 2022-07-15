import torch

import numpy as np

import os
import yaml
from pathlib import Path
from distutils.dir_util import copy_tree
from types import SimpleNamespace   
import time

from utils import check_paths

from setup import get_experiment, get_data_id, get_model, get_model_unc, get_optim

from plotting import plot_curves_advanced


def train(model_unc, trainloader, optimizer, criterion, scheduler=None, device='cuda:0', epochs=10, start_epoch=0,
          gradient_penalty=0, save_every=5, save_dir=None, exp_name='test', save_model=False, alpha=0.0):
    r"""
    Train a given uncertainty model and dataset with the provided settings and return the loss list and accuracy list. 
    The model is also saved during training in the given directory. This function takes into account specific functionalities
    that are related to the uncertainty model such as updating the model at the end and saving the model and uncertainty
    model seprately.
    """
    
    model_unc.to(device)
    model_unc.train()
    
    loss_list = []
    accuracy_list = []
    
    start_time = time.time()
    
    for epoch in range(start_epoch, start_epoch+epochs): 
        
        criterion.epoch = epoch  # Not necessarily used by every method
        
        total_loss = 0
        total = 0
        correct = 0
        
        for i, (imgs, targets) in enumerate(trainloader):
            
            imgs, targets = imgs.to(device), targets.to(device)
            
            model_unc.zero_grad()
            optimizer.zero_grad()
            
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
                perm = torch.randperm(imgs.shape[0], generator=trainloader.sampler.generator).to(device)
                
                imgs = lam * imgs + (1-lam) * imgs[perm]
                targets_a, targets_b = targets, targets[perm] 

            if gradient_penalty > 0:
                imgs.requires_grad_(True)           
            
            logits = model_unc(imgs)
            
            if alpha > 0:
                loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            else:
                loss = criterion(logits, targets)
            
            if gradient_penalty > 0:
                loss += gradient_penalty * calc_gradient_penalty(imgs, logits)
                imgs.requires_grad_(False)
            
            loss.backward()

            optimizer.step()
            
            model_unc.update(imgs, targets)  # Does not necessarily do something for every method
            
            total_loss += float(loss)
            total += float(logits.shape[0]) if not isinstance(logits, tuple) else logits[0].shape[0]
            
            if len(targets.shape) > 1:
                targets = torch.argmax(targets, dim=1)
                
            correct += float(torch.sum((torch.argmax(logits, dim=1) == targets), dim=0)) if not isinstance(logits, tuple) else float(torch.sum((torch.argmax(logits[0], dim=1) == targets), dim=0))
            
        average_loss = total_loss / len(trainloader)
        accuracy = correct / total
        loss_list.append(average_loss)
        accuracy_list.append(accuracy)
        
        total_time = time.time() - start_time
        time_all = (np.round(total_time, 3), np.round(total_time / 60, 3), np.round(total_time / 3600, 3))  # s, min, h
        
        print(f'epoch: {str(epoch):3s} => loss: {str(average_loss):22s}, average train accuracy: {str(accuracy):10s}, time: {str(time_all[0]):6s}s, {str(time_all[1]):6s}m, {str(time_all[2]):6s}h')
        
        if epoch % save_every == 0 and save_dir is not None or epoch == (epochs - 1):
            
            if save_model:
                torch.save(model_unc.model.state_dict(), save_dir + exp_name + f'_{epoch}.pth')
            else:
                torch.save(model_unc.state_dict(), save_dir + exp_name + f'_uncertain_{epoch}.pth')
            
        if scheduler is not None:
            scheduler.step()
            
    return model_unc, loss_list, accuracy_list

def calc_gradient_penalty(x, y_pred):
    r"""
    Calculates the gradient penalty for a given set of outputs and inputs.
    """

    # We take the gradients of the predictions w.r.t. the inputs. Note that summing y_pred gives the exact same results
    # Note that we set create_graph to True as as it makes backpropagation through the gradient operation possible 
    # We also set retain graph to True as it makes sure not te remove the current graph as we want to reuse it again
    # when calling backprop on the loss. It seems to be the case that when not specifying retain_graph i.e. keeping it
    # at the basic value which is None but setting create_graph=True, that it will also retain the graph
    gradients = torch.autograd.grad(outputs=y_pred, inputs=x, grad_outputs=torch.ones_like(y_pred), create_graph=True, retain_graph=True)[0]
    
    # Flatten the gradients to obtain a vector of -> (bs, c*h*w)
    gradients = gradients.flatten(start_dim=1)

    # Obtain the l2 norm of the vector to obtain -> (bs,)
    grad_norm = torch.linalg.norm(gradients, ord=2, dim=1)  # same as gradients.norm(2, dim=1)

    # Calculate the gradient penalty and average over the batch to obtain -> (1,)
    gradient_penalty = torch.mean((grad_norm - 1) ** 2, dim=0)

    return gradient_penalty


def run(data_dir, save_dir, save_dir_par, config, seed, exp_name, device):
    """Performs training according to the specified settings in the config file"""
    
    print(f'\nSTARTED EXPERIMENT: {exp_name}')
    config = SimpleNamespace(**config.main_run)

    print("\nCREATING THE SAVE DIRECTORY")
    save_dir_eval = save_dir + 'state_dicts/'
    Path(save_dir_eval).mkdir(parents=True, exist_ok=True)
    print(f'saving data to {save_dir_eval}')

    print("\nLOADING THE ID DATA")
    trainloader, _, num_classes = get_data_id(config, data_dir, seed, device, train=True)

    print("\nLOADING THE MODEL AND UNCERTAINTY MODEL")
    model = get_model(config, num_classes)
    model_unc = get_model_unc(model, config, num_classes, trainloader=trainloader)

    if config.epochs > 0:
        print("\nLOADING THE OPTIMIZATION PARAMETERS")
        optimizer, criterion, scheduler = get_optim(model_unc, config)

        print("\nSTARTED TRAINING")
        model_unc, loss_list, accuracy_list = train(model_unc, trainloader, optimizer, criterion, scheduler, device=device, 
                                                    epochs=config.epochs, start_epoch=config.start_epoch, 
                                                    gradient_penalty=config.gradient_penalty, save_every=config.save_every, 
                                                    save_dir=save_dir_eval, exp_name=exp_name, save_model=config.save_model, 
                                                    alpha=config.alpha)

        print("\nFINISHED TRAINING, PLOTTING CURVES")
        data = [ [(np.arange(1,len(loss_list)+1), loss_list, None, config.model)], [(np.arange(1,len(loss_list)+1), accuracy_list, None, config.model)] ]
        format = [ ['epoch', 'loss', None, 'loss curve', None], ['epoch', 'train accuracy', None, 'accuracy curve', None] ]
        training_curves = plot_curves_advanced(data, format, suptitle='training curves')
        training_curves.savefig(save_dir_eval + 'training_curves.png')
    
    print("\nLOADING THE ID DATA")
    trainloader, _, num_classes = get_data_id(config, data_dir, seed, device, train=False)

    print('\nSTARTED POST PROCESSING')
    model_unc.post_proces(trainloader, device)
    print('\nFINISHED POST PROCESSING')

    torch.save(model_unc.state_dict(), save_dir_eval + exp_name + '_uncertain_post.pth')

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
            # config_files = sorted([file for file in os.listdir(experiment.config_files)])

    print('\nTRAINING ON FILES')
    print(config_files)
    check_paths(experiment.configs_dir, config_files)

    for config in config_files:

        experiment.config_file = config
        data_dir, save_dir, save_dir_par, config, seed, exp_name, device = get_experiment(experiment, return_dirs=False)

        run(data_dir, save_dir, save_dir_par, config, seed, exp_name, device)

    

    
