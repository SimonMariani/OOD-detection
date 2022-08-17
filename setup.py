import os
    
import torch
import torchvision
import torchvision.transforms as transforms
import custom_transforms as transforms_custom
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import ConcatDataset

import numpy as np
from types import SimpleNamespace   

import random

from dataset import notMNIST, DatasetFromSubset
from networks import LeNet, ResNet

from models_uncertainty import BasicModel, DPN, EDL, DUQ, Mahalanobis, ODIN, Gram, Ensemble
from losses import DPNKlLoss, EDLMSELoss

import yaml
import pickle

from utils import make_result_directory, seed_everything, overwrite_built_in, normalize, get_inverse_normalization
from torchvision.transforms.transforms import InterpolationMode

import ast


### The default setup ###

def get_experiment(experiment='experiment.yaml', return_dirs=False):
    
    if isinstance(experiment, str):  # if it's a string we load else it's already a namespace
        with open(experiment) as f:
            experiment = yaml.load(f, yaml.SafeLoader)
            experiment = SimpleNamespace(**experiment)

    data_dir = experiment.data_dir
    configs_dir = experiment.configs_dir
    run_dir = experiment.run_dir
    results_dir = experiment.results_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    config_file = experiment.config_file
   
    with open(configs_dir + config_file) as f:
        config = yaml.load(f, yaml.SafeLoader)
        config = SimpleNamespace(**config)

    save_dir = make_result_directory(run_dir, config.exp_name, configs_dir, config_file, use_timestamp=True)
    save_dir_par = make_result_directory(results_dir, config.exp_name, configs_dir, config_file, use_timestamp=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    overwrite_built_in()
    seed_everything(config.seed, set_deter_algo=True)
    
    if not return_dirs:
        return data_dir, save_dir, save_dir_par, config, config.seed, config.exp_name, device
    else:
        return data_dir, configs_dir, run_dir, results_dir, save_dir, save_dir_par, config_file, config, config.seed, config.exp_name, device


### Data setup functions ###

def get_data_id(config, data_dir='/mnt/hdd/data/', seed=42, device='cuda:0', train=True):
    r"""
    Returns dataset objects for all the specified datasets, it also returns the number of classes corresponding to
    the training data.
    """
    
    traindataset_ood = None
    
    num_classes = get_num_classes(config.dataset)
    transform_train, transform_test = get_data_transform(dataset=config.dataset, config=config)
    target_transform, target_transform_out = get_target_transform(config, num_classes)

    if train:
        batch_size = config.batch_size if hasattr(config, 'batch_size') else 256
        randomize = True
    else:
        transform_train = transform_test
        batch_size = 256  # TODO write max batch size function if possible
        randomize = False
       
    traindataset, testdataset = get_pytorch_dataset(config.dataset, data_dir, transform_train, transform_test, 
                                                    target_transform=target_transform, seed=seed)
    print(f'loaded {config.dataset} as ID dataset')

    if config.dataset_ood_train is not None:  # add an additional ood dataset as training set 
        traindataset_ood, _ = get_pytorch_dataset(config.dataset_ood_train, data_dir, transform_train, transform_test, 
                                                  target_transform=target_transform_out, seed=seed)
        traindataset = ConcatDataset([traindataset, traindataset_ood])
        print(f'loaded {config.dataset_ood_train} as training OOD dataset')

    if config.weight_path is not None:  # add weights per sample
        with open(config.weight_path, 'rb') as handle:
            _, _, weights = pickle.load(handle) 

            weights = 1 - normalize([weights])[0] if config.invert else normalize([weights])[0]
            traindataset.weights = weights
        print("added weights to the dataset")

    if config.class_weights is not None:  # add weights per class
        class_weights = torch.ones((len(traindataset.targets)))
        for label, weight in enumerate(config.class_weights):
            class_weights[traindataset.targets == label] = weight
        traindataset.class_weights = class_weights
        print("added class weights")
    
    trainloader = get_data_loader(traindataset, batch_size, randomize=randomize, seed=seed, device=device)
    testloader = get_data_loader(testdataset, 256, randomize=False, seed=seed, device=device) 

    return trainloader, testloader, num_classes

def get_data_ood(config, datasets=[None], data_dir='/mnt/hdd/data/', seed=42, device='cuda:0'):
    r"""
    Returns dataset objects for all the specified datasets, it also returns the number of classes corresponding to
    the training data.
    """

    _, transform_test = get_data_transform(dataset=config.dataset, config=config)
    
    testloaders_ood = []
    for dataset in datasets:
        _, testdataset_ood = get_pytorch_dataset(dataset, data_dir, transform_test, transform_test, target_transform=None, seed=seed)
        testloaders_ood.append(get_data_loader(testdataset_ood, 256, randomize=False, seed=seed, device=device))

        print(f'loaded dataset {dataset} as ood data')

    return testloaders_ood


def get_num_classes(dataset):
    r"""
    Returns the number of classes for a given dataset.
    """
    
    if dataset in ['mnist', 'fashion_mnist', 'notmnist', 'cifar10', 'svhn']:
        return 10
    elif dataset in ['cifar100']:
        return 100
    elif dataset in ['imagenet', 'imagenet_r', 'imagenet_c']:
        return 1000
    elif dataset in ['omniglot']:
        return 1623
    else:
        raise ValueError(f'invalid dataset: {dataset}')
        
def get_data_transform(dataset, config):
    r"""
    Returns the data transforms that will be used inside the dataloader.
    """
    
    if dataset in ['mnist', 'fashion_mnist', 'notmnist', 'omniglot']:
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        
    elif dataset in ['cifar10', 'cifar100', 'svhn']:
        transform_train = transforms.Compose([transforms.RandomCrop((32, 32), padding=4), transforms.ToTensor(),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), 
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
    elif dataset in ['imagenet', 'tiny_imagenet', 'imagenet_r', 'imagenet_c']: 
        transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    else:
        raise ValueError(f'data transform for {dataset} unavailable')

    if config.data_transform is not None:

        for transform in config.data_transform:

            if transform == 'colorjitter':
                transform_train.transforms.insert(-1, transforms.ColorJitter(brightness=1, contrast=3, saturation=3, hue=0.5))

            elif transform == 'rotate':
                transform_train.transforms.insert(-1, transforms.RandomRotation(180))

            elif transform == 'erase':
                transform_train.transforms.insert(-1, transforms.RandomErasing(p=0.5))

            elif transform == 'perspective':
                transform_train.transforms.insert(-1, transforms.RandomPerspective(p=0.5))

            elif transform == 'equalize':
                transform_train.transforms.insert(1, transforms.RandomEqualize(p=1)) 

            elif transform == 'affine':
                transform_train.transforms.insert(-1, transforms.RandomAffine(180)) 

            elif transform == 'solarize':
                transform_train.transforms.insert(-1, transforms.RandomSolarize(0))  

            elif transform == 'blur':
                transform_train.transforms.insert(-1, transforms.GaussianBlur(5))   

            elif transform == 'augmentpolicy':
                if dataset == 'cifar10':
                    auto_augment_policy = transforms.autoaugment.AutoAugmentPolicy.CIFAR10 
                elif dataset == 'svhn':
                    auto_augment_policy = transforms.autoaugment.AutoAugmentPolicy.SVHN
                else:
                    auto_augment_policy = transforms.autoaugment.AutoAugmentPolicy.IMAGENET

                transform_train.transforms.insert(1, transforms.AutoAugment(policy=auto_augment_policy))  
    

    print(f'loaded train transform: {transform_train}')
    print(f'loaded test transform: {transform_test}')

    return transform_train, transform_test

def get_target_transform(config, num_classes):
    r"""
    Returns the label transform for a given method, specifically some methods use one hot labels and cast this one hot 
    encoding as a distribution.
    """
    
    if config.target_transform == 'one_hot':
        target_transform = transforms.Compose([transforms_custom.OneHotEncoding(num_classes)])
        target_transform_out = transforms.Compose([transforms_custom.UniformLabels(num_classes, size=1)])

    elif config.target_transform == 'smooth_one_hot':
        target_transform = transforms.Compose([transforms_custom.SmoothOneHotEncoding(num_classes, precision=config.precision)])
        target_transform_out = transforms.Compose([transforms_custom.UniformLabels(num_classes, size=1)])

    else:
        target_transform = None
        target_transform_out = None
    
    return target_transform, target_transform_out


def get_pytorch_dataset(dataset, data_dir='/mnt/hdd/data/', transform_train=None, transform_test=None, target_transform=None, 
                        seed=42, split=[0.8, 0.2], ):
    r"""
    Returns a dataset as a PyTorch Dataset object. Note that some datasets can be downloaded automatically via PyTorch while
    others need to be downloaded manually and put into the right folder.
    In the case that a dataset does not have a train and test set a random split will be made of which the sizes are 
    determined by the split argument.
    """

    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
        
    elif dataset == 'fashion_mnist':
        trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True)
        
    elif dataset == 'notmnist':
        fullset = notMNIST(root=data_dir + 'notMNIST_small', class_samples=None) #'notMNIST/'
        trainset, testset = random_split(fullset, split=split, seed=seed)
        
    elif dataset == 'omniglot':
        trainset = torchvision.datasets.Omniglot(root=data_dir, background=True, download=True)
        testset = torchvision.datasets.Omniglot(root=data_dir, background=False, download=True)
        
    elif dataset == 'cifar10':  # Test set is used for ood detection
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    elif dataset == 'cifar100':  # Test set is used for ood detection
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
        
    elif dataset == 'svhn':  # Test set is used for ood detection
        trainset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True)
        testset = torchvision.datasets.SVHN(root=data_dir, split='test', download=True)
        
    elif dataset == 'sun':  # Use 10000 randomly sampled samples as test set
        fullset = torchvision.datasets.SUN397(root=data_dir, download=False)
        split = [(len(testset)-10000) / len(testset), 10000/len(testset)]
        trainset, testset = random_split(fullset, split=split, seed=seed)
    
    elif dataset == 'lsun': # Test set is used for ood detection
        trainset = torchvision.datasets.LSUN(root=data_dir + 'lsun/', classes='train', transform=transform_train, target_transform=target_transform) 
        #valset = torchvision.datasets.LSUN(root=data_dir + 'lsun/', classes='val', transform=transform_test, target_transform=target_transform) 
        testset = torchvision.datasets.LSUN(root=data_dir + 'lsun/', classes='test', transform=transform_test, target_transform=target_transform)
    
    elif dataset == 'lsun_nearest':  # Test set is used for ood detection even if it does not have any labels
        old_resize = transform_test.transforms[0]  # we make the assumption that the first transform is always a resize
        new_resize = transforms.Resize(size=old_resize.size, interpolation=InterpolationMode.NEAREST, max_size=old_resize.max_size,
                        antialias=old_resize.antialias)

        transform_test = transforms.Compose([new_resize] + transform_test.transforms[1:])

        trainset = torchvision.datasets.LSUN(root=data_dir + 'lsun/', classes='train', transform=transform_train, target_transform=target_transform) 
        testset = torchvision.datasets.LSUN(root=data_dir + 'lsun/', classes='test', transform=transform_test, target_transform=target_transform)

    elif dataset == 'lsun_bicubic':  # Test set is used for ood detection even if it does not have any labels
        old_resize = transform_test.transforms[0]  # we make the assumption that the first transform is always a resize
        new_resize = transforms.Resize(size=old_resize.size, interpolation=InterpolationMode.BICUBIC, max_size=old_resize.max_size,
                        antialias=old_resize.antialias)

        transform_test = transforms.Compose([new_resize] + transform_test.transforms[1:])

        trainset = torchvision.datasets.LSUN(root=data_dir + 'lsun/', classes='train', transform=transform_train, target_transform=target_transform) 
        testset = torchvision.datasets.LSUN(root=data_dir + 'lsun/', classes='test', transform=transform_test, target_transform=target_transform)

    elif dataset == 'lsun_odin':  # Test set is used for ood detection even if it does not have any labels
        trainset = torchvision.datasets.ImageFolder(root=data_dir + "LSUN_resize")
        testset = torchvision.datasets.ImageFolder(root=data_dir + 'LSUN_resize')

    elif dataset == 'places':  # Val set is used for ood detection (although a lot larger than before) (also no test set available)
        trainset = torchvision.datasets.Places365(root=data_dir + 'places/', split='train-standard', small=True, download=False, transform=transform_train, target_transform=target_transform)
        testset = torchvision.datasets.Places365(root=data_dir + 'places/', split='val', small=True, download=False, transform=transform_test, target_transform=target_transform)

    elif dataset == 'imagenet':  # If used for ood detection the val split is used
        trainset = torchvision.datasets.ImageFolder(root=data_dir + 'ImageNet/train')
        testset = torchvision.datasets.ImageFolder(root=data_dir + 'ImageNet/val')

    elif dataset == 'inaturalist':  # val set is used for ood detection (also bigger and no train split available)
        trainset = torchvision.datasets.INaturalist(root=data_dir + 'inaturalist', version='2021_train', download=False)
        testset = torchvision.datasets.INaturalist(root=data_dir + 'inaturalist', version='2021_valid', download=False)  

    elif dataset == 'tiny_imagenet':  # Test set is used for ood detection even if it does not have any labels
        trainset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/train')

        # We need to use the validation set as we have semantically coherent labels for those, this is however inconsistent
        # with other methods that are generally evaluated on the test set. Note that we do need the test data to compare to the ODIN data
        testset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/val') 
        # testset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/test')

    elif dataset == 'tiny_imagenet_nearest':  # Test set is used for ood detection even if it does not have any labels
        trainset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/train')
        testset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/val')  
        # testset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/test')

        old_resize = transform_test.transforms[0]  # we make the assumption that the first transform is always a resize
        new_resize = transforms.Resize(size=old_resize.size, interpolation=InterpolationMode.NEAREST, max_size=old_resize.max_size,
                        antialias=old_resize.antialias)

        transform_test = transforms.Compose([new_resize] + transform_test.transforms[1:])

    elif dataset == 'tiny_imagenet_bicubic':  # Test set is used for ood detection even if it does not have any labels
        trainset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/train')
        testset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/val')  
        # testset = torchvision.datasets.ImageFolder(root=data_dir + 'tiny-imagenet-200/test')

        old_resize = transform_test.transforms[0]  # we make the assumption that the first transform is always a resize
        new_resize = transforms.Resize(size=old_resize.size, interpolation=InterpolationMode.BICUBIC, max_size=old_resize.max_size,
                        antialias=old_resize.antialias)

        transform_test = transforms.Compose([new_resize] + transform_test.transforms[1:])

    elif dataset == 'tiny_imagenet_odin':  # Test set is used for ood detection even if it does not have any labels
        trainset = torchvision.datasets.ImageFolder(root=data_dir + "Imagenet_resize")
        testset = torchvision.datasets.ImageFolder(root=data_dir + 'Imagenet_resize')
        
    elif dataset == 'imagenet_r': 
        fullset = torchvision.datasets.ImageFolder(root=data_dir + 'imagenet-r/')
        trainset, testset = random_split(fullset, split=split, seed=seed)
        
    elif dataset.split('+')[0] == 'imagenet_c': 
        splitted = dataset.split('+')
        subpath = splitted[1] + '/' + splitted[2] + '/' + splitted[3] + '/'
        fullset = torchvision.datasets.ImageFolder(root=data_dir + 'imagenet-c/' + subpath)
        trainset, testset = random_split(fullset, split=split, seed=seed)

    elif dataset == 'textures': # The entire dataset is used for ood detection (generally no training takes place on this dataset)
        trainset_temp = torchvision.datasets.DTD(root=data_dir, split='train', download=True)
        valset_temp = torchvision.datasets.DTD(root=data_dir, split='val', download=True)
        testset_temp = torchvision.datasets.DTD(root=data_dir, split='test', download=True)

        trainset = DatasetFromSubset(ConcatDataset([trainset_temp, valset_temp, testset_temp]))
        testset = DatasetFromSubset(ConcatDataset([trainset_temp, valset_temp, testset_temp]))

    else:
        raise ValueError(f'dataset {dataset} unavailable')
    
    # We only directed passed the transforms to the dataset where it was necessary to pass immediately
    trainset.transform, trainset.target_transform  = transform_train, target_transform
    testset.transform, testset.target_transform = transform_test, target_transform
    
    return trainset, testset

def random_split(dataset, split=[0.8, 0.2], seed=42):
    r"""
    Randomly splits the dataset into multiple segments and then again creates datasets as the PyTorch subset object lacks
    some properties.
    """
    
    dataset_size = len(dataset)
    split_num_samples = []
    for val in split:
        split_num_samples.append(int(round(dataset_size*val)))
        
    splitted_datasets = torch.utils.data.random_split(dataset, split_num_samples,  
                                                     generator=torch.Generator().manual_seed(seed))
    
    splitted_datasets = [DatasetFromSubset(splitted_dataset) for splitted_dataset in splitted_datasets]
        
    return splitted_datasets


def get_data_loader(dataset, batch_size, randomize=False, seed=42, replacement=False, device='cuda:0'):
    r"""
    Returns a PyTorch Dataloader object which can either be (deterministically) randomized with or without class
    weights, or not randomized which will then always be in the same order.
    Note that the dataloader uses a worker_init_fn which is provided by the PyTorch docs to ensure deterministic
    behavior when generating batches.
    """
    
    if randomize:
        device = device if hasattr(dataset, 'weights') or hasattr(dataset, 'class_weights') else 'cpu' 
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        
        if hasattr(dataset, 'weights'):
            dataset.weights = dataset.weights.to(device)
            rand_sampler = torch.utils.data.WeightedRandomSampler(dataset.weights, len(dataset), generator=generator)
            print("added weights to dataloader")
        elif hasattr(dataset, 'class_weights'):
            dataset.class_weights = dataset.class_weight.to(device)
            rand_sampler = torch.utils.data.WeightedRandomSampler(dataset.class_weights, len(dataset), generator=generator)
            print("added class weights to dataloader")
        else:
            rand_sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement, generator=generator)
        
    else:
        rand_sampler = None
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, sampler=rand_sampler, 
                                             shuffle=False, worker_init_fn=seed_worker)
    
    return dataloader

def seed_worker(worker_id):
    r"""
    The worker_init_fn provided by the PyTorch Docs as to provide deterministic behavior. This function is passes as the worker_init_fn 
    argument when calling the DataLoader class.
    """
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


### Model setup functions ###

def get_model(config, num_classes):
    r"""
    Returns a model and automatically changes the input and output sizes. It also automatically changes the input and 
    output sizes of the model and freezes the layers as specified.
    """

    num_classes = config.model_out_size if config.model_out_size is not None else num_classes
    
    if config.dataset in ['mnist', 'fashion_mnist', 'notmnist', 'omniglot']:
        in_size = (1, 28)
    
    elif config.dataset in ['cifar10', 'cifar100', 'svhn']:
        in_size = (3, 32)
    
    elif config.dataset in ['imagenet', 'imagenet_r', 'imagenet_c']:
        in_size = (3, 224)
    
    else:
        raise ValueError(f'invalid dataset: {config.dataset}')
            
    if config.model == 'lecun':
        model = LeNet()
        orig_in_size, orig_out = (3, 32), 10
    
    elif config.model in ['resnet18', 'resnet34', 'resnet50']:
        model = ResNet(resnet=config.model, pretrained=config.res_pre, progress=False, max_pool=config.max_pool, small_conv=config.small_conv)
        orig_in_size, orig_out = (3, 224), 1000
        orig_in_size = (3, in_size[1])
    
    else:
        raise ValueError(f'model {config.model} unavailable')
    
    print(f'loaded model {config.model}')
    
    if in_size != orig_in_size:
        model.change_input_size(in_size=in_size, orig_in_size=orig_in_size)
        print("changed model input size")
    
    if orig_out != num_classes:
        model.change_output_size(num_classes)
        print("changed model output size")
        
    if config.pretrained_path is not None:
        weight_dict = torch.load(config.pretrained_path)

        # Change the names of some keys, does not always do something but is necessary for the use of some pretrained models
        weight_dict = {key.replace('shortcut', 'downsample') : val for key, val in weight_dict.items() }
        weight_dict = {key.replace('linear', 'fc') : val for key, val in weight_dict.items() }

        model.load_state_dict(weight_dict)
        print(f'loaded pretrained model')
        
    model.freeze_until(config.freeze_until)
    print(f'froze layers until layer {config.freeze_until}')
        
    return model

def get_model_unc(model, config, num_classes, trainloader=None, experiment='experiment.yaml'):
    r"""
    Returns an uncertainty model s.t. the given model is integrated in it. It also passes all the necessary parameters
    to the uncertainty model and loads the pretrained model if available.
    """

    data_sample = trainloader.dataset[0][0].unsqueeze(0).repeat(2,1,1,1)

    if config.model_unc == 'basic':
        model_unc = BasicModel(model=model, num_classes=num_classes, data_sample=data_sample, unc_measure=config.unc_measure)
    
    elif config.model_unc == 'maha':
        model_unc = Mahalanobis(model=model, num_classes=num_classes, data_sample=data_sample, cov_type=config.cov_type, 
                                num_batches=config.num_batches, print_every=config.print_every, use_layers=config.use_layers, 
                                retain_info=config.retain_info, unc_measure=config.unc_measure, compress=config.compress)
        
    elif config.model_unc == 'gram':
        model_unc = Gram(model, num_classes=num_classes, data_sample=data_sample, power=config.power, num_batches=config.num_batches, 
                        print_every=config.print_every, use_layers=config.use_layers, retain_info=config.retain_info, 
                        unc_measure=config.unc_measure, compress=config.compress)

    elif config.model_unc == 'odin':
        criterion = get_criterion(config)
        _, _, std = get_inverse_normalization(trainloader.dataset, return_vals=True)
        model_unc = ODIN(model=model, num_classes=num_classes, data_sample=data_sample, 
                        temperature=config.temperature, criterion=criterion, magnitude=config.magnitude, data_std=std)
    
    elif config.model_unc == 'dpn':
        model_unc = DPN(model=model, num_classes=num_classes, data_sample=data_sample)
        
    elif config.model_unc == 'edl':
        model_unc = EDL(model=model, num_classes=num_classes, data_sample=data_sample)
        
    elif config.model_unc == 'duq':
        model_unc = DUQ(model=model, num_classes=num_classes, data_sample=data_sample,
                       embedding_size=config.embedding, model_out_size=config.model_out_size, length_scale=config.length_scale,
                       learn_length_scale=config.learn_length_scale, gamma=config.gamma)
    
    elif config.model_unc == 'ensemble':

        # Open the universal experiment
        with open(experiment) as f:
            experiment = yaml.load(f, yaml.SafeLoader)
            experiment = SimpleNamespace(**experiment)

        # Loop over the configuration to include in the ensemble and their pretrained paths
        all_models = []
        for i, config_temp in enumerate(config.configs):
            
            # Get the experiment settings for the given config
            experiment.config_file = config_temp
            data_dir, save_dir, save_dir_par, config_temp, seed, exp_name, device = get_experiment(experiment)

            config_temp = SimpleNamespace(**config_temp.main_run)

            # Change the path to the post model
            config_temp.pretrained_path_unc = save_dir_par + 'state_dicts/' + exp_name + '_uncertain_post.pth'

            all_models.append(get_model_unc(model, config_temp, num_classes, data_sample, trainloader))

        model_unc = Ensemble(model=all_models, num_classes=num_classes, data_sample=data_sample)

    else:
        raise ValueError(f'uncertainty model {config.model_unc} unavailable')
    
    print(f'loaded uncertainty model {config.model_unc}')

    if config.pretrained_path_unc is not None and config.pretrained_path_unc != 'ignore':
        model_unc.load_state_dict(torch.load(config.pretrained_path_unc))
        print(f'loaded pretrained uncertainty model from {config.pretrained_path_unc}')
    
    return model_unc


### Optimization setup functions ###

def get_optim(model, config):
    r"""
    Returns the optimizer, criterion and scheduler given a specif method and possible additional hyperparameters
    """
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = get_optimizer(params, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = get_criterion(config)
        
    return optimizer, criterion, scheduler

def get_criterion(config):
    
    if config.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        
    elif config.criterion == 'dpn_kl':
        criterion = DPNKlLoss(reduction='mean')
        
    elif config.criterion == 'edl_mse':
        criterion = EDLMSELoss(annealing_step=config.annealing_step, reduction='mean')
        
    elif config.criterion == 'bce':
        criterion = nn.BCELoss(reduction='mean')

    elif config.criterion == 'mse':
        criterion = nn.MSELoss(reduction='mean' ) 
    else:
        raise ValueError(f'loss for {config.criterion} is not available')   
    
    print(f'loaded criterion {config.criterion}')

    return criterion

def get_optimizer(params, config):
    
    if config.optimizer == 'sgd': # default: momentum=0, dampening=0, weight_decay=0, nesterov=False
        optimizer = optim.SGD(params, lr=config.lr, momentum=config.momentum, dampening=config.dampening, 
                              weight_decay=config.weight_decay, nesterov=config.nesterov)
        
    elif config.optimizer == 'adam':  # default: lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
        optimizer = optim.Adam(params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    
    else:
        raise ValueError(f'optimizer {config.optimizer} is not available')
    
    print(f'loaded optimizer {config.optimizer}')

    return optimizer

def get_scheduler(optimizer, config):
    
    if config.scheduler == 'cosine_annealing_lr':  # default: eta_min=0, last_epoch=- 1, verbose=False
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.t_max)
        
    elif config.scheduler == 'multi_step_lr':  # default: gamma=0.1, last_epoch=- 1, verbose=False
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma) 
    
    elif config.scheduler == 'step_lr':  # default: gamma=0.1, last_epoch=- 1, verbose=False
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    
    elif config.scheduler is None:
        scheduler = None
        
    else:
        raise ValueError(f'Scheduler {config.scheduler} is not available')
        
    print(f'loaded scheduler {config.scheduler}')

    return scheduler
    


### Loading saved data setup functions ###

def load_data(config, datasets_ood, save_dir_par, exp_name):

    id_data_path = save_dir_par + 'inference/' + exp_name +  f'_{config.dataset}.pickle'
    with open(id_data_path, 'rb') as handle:
        logits, targets, uncertainty = pickle.load(handle) 
    print(f'loaded id data from {id_data_path}')

    data_ood_full = {}
    for dataset_ood in datasets_ood:
        ood_data_path = save_dir_par + 'inference/' + exp_name +  f'_{dataset_ood}.pickle'

        with open(ood_data_path, 'rb') as handle:
            logits_ood, targets_ood, uncertainty_ood = pickle.load(handle)
            data_ood_full[dataset_ood] = [logits_ood, targets_ood, uncertainty_ood]
        print(f'loaded ood data from {ood_data_path}')

    return logits, targets, uncertainty, data_ood_full


def load_alt_labels(config, datasets_ood, testloaders_ood, data_dir, device):
    alternate_labels = {}
    path = data_dir + f'imglist/benchmark_{config.dataset}/'

    for i, (dataset_ood, testloader_ood) in enumerate(zip(datasets_ood, testloaders_ood)):
        path_test = path + f'test_{dataset_ood}.txt'
        
        if os.path.exists(path_test):
            with open(path_test) as imgfile:
                imglist_raw = imgfile.readlines()
                
            imglist = [ast.literal_eval(line.strip("\n").split(' ', 1)[1])['sc_label'] for line in imglist_raw]
            files = [line.strip("\n").split(' ', 1)[0].split('/')[-1] for line in imglist_raw]
            
            imglist = match_targets(dataset_ood, testloader_ood, torch.tensor(imglist).to(device), files)
            alternate_labels[dataset_ood] = (imglist, files)
        
        else:
            alternate_labels[dataset_ood] = (None, None)
        
        print(f'loaded alternate labels from {path}')

    return alternate_labels

def match_targets(name, testloader_ood, targets_alt, filenames_alt):
    
    if name == 'tiny_imagenet':
        filenames_dataset = [name.split('/')[-1] for name, _ in testloader_ood.dataset.imgs]
        indices = [int(name.split('.')[0].split('_')[-1]) for name in filenames_dataset]
        targets_alt = targets_alt[indices]
        
    elif name == 'lsun':        
        filenames_alt = [filename.split('.')[0].encode('UTF-8') for filename in filenames_alt]
        targets_alt = [target for _, target in sorted(zip(filenames_alt, targets_alt))]
        targets_alt = torch.tensor(targets_alt)
        
    elif name == 'places':
        targets_alt = [target for _, target in sorted(zip(filenames_alt, targets_alt))]
        targets_alt = torch.tensor(targets_alt)
        
    else:
        targets_alt = None
    
    return targets_alt