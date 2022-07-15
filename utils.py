import torch
import torchvision

import os
from datetime import datetime
from pathlib import Path
import shutil

import random
import pickle
import numpy as np

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.utils import verify_str_arg, iterable_to_str

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset

from PIL import Image

import string
import io



def seed_everything(seed, set_deter_algo=False):
    r"""
    A seeding function that ensures deterministic behavior throughout all of the experiments. Note that some other
    functionalities requires the seed to be set there and that when using a dataloader the worker initialization
    requires a specific
    """
    
    # Standard randomized processes by python and numpy
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch randomized processes
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional PyTorch settings to ensure deterministic behavior
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Set Torch to unly use deterministic algorithms and change the environmental variable (this may hurt performance)
    if set_deter_algo:
        #torch.use_deterministic_algorithms(True)
        torch.use_deterministic_algorithms(mode=True, warn_only=True)

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    # add an environmental variable
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize(all_data):
    r"""
    Standard minmax normalization on multiple datasets where the min and max values are extracted from the entire 
    data pool.
    """
    temp = torch.concat(all_data, dim=0)
    max_val, _ = torch.max(temp, dim=0)
    min_val, _ = torch.min(temp, dim=0)
    
    all_data_norm = []
    for data in all_data:
        all_data_norm.append( (data - min_val) / (max_val - min_val) )
    
    return all_data_norm


def get_inverse_normalization(dataset, return_vals=False):
    
    if isinstance(dataset, torch.utils.data.ConcatDataset):  # In the case of concatenated datasets we date the first dataset to take the transforms from
        dataset = dataset.datasets[0]

    if isinstance(dataset, torch.utils.data.Subset):  # In the case of a subset we take the dataset 
        dataset = dataset.dataset

    mean, std = dataset.transform.transforms[-1].mean, dataset.transform.transforms[-1].std
    
    if not (isinstance(mean, tuple) or isinstance(mean, list)):
        mean, std = [mean], [std]
        
    inverse_norm = transforms.Compose([transforms.Normalize(mean=[0.0 for val in mean], std=[1/val for val in std]),
                                       transforms.Normalize(mean=[-val for val in mean], std=[1.0 for val in std]),
                                       torch.nn.ReLU()])
    
    if return_vals:
        return inverse_norm, mean, std
    
    return inverse_norm


def make_result_directory(run_dir, exp_name, configs_dir, config_file, use_timestamp=True):
    r"""
    This method creates am unique directory based on a configuration and returns the director name
    """

    if use_timestamp:
        # Convert the timestamp to a suitable string s.t. we can have a unique directory for every run
        timestamp = datetime.now()
        timestamp_string = str(timestamp).replace(" ", "_").replace(".", "_").replace(":", "-")

        # Define the pathname and make the path
        save_dir = run_dir + exp_name + '/' + timestamp_string + '/'
    else:
        save_dir = run_dir + exp_name + '/'

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Copy the config file to this directory
    original = configs_dir + config_file
    target = save_dir + config_file
    shutil.copyfile(original, target)

    return save_dir


def make_pos_def(G):
    r"""This function takes non-positive definite matrix and makes it positive definite using the method described in 
    https://animalbiosciences.uoguelph.ca/~lrs/ELARES/PDforce.pdf. However, it does not always seem to work and there
    was no mathematical proof in the paper."""

    # TODO See if we want to keep this function as it is not used, it is however, a very cool function.

    D, U = torch.linalg.eig(G)
    D, U = D.float(), U.float()

    D_neg = D[D<0]
    D_pos = D[D>=0]

    s = (torch.sum(D_neg, dim=0) * 2)
    t = s**2 * 100 + 1
    p = torch.amin(D_pos)

    D_neg_star = p * (s - D_neg) * (s - D_neg) / t 
    D[D<0] = D_neg_star

    G_star = U @ torch.diag(D) @ U.T

    return G_star


def check_paths(directory, filenames):

    for file in filenames:

        if not os.path.isfile(directory + file):
            raise ValueError(f'the file {directory + file} is not a correct path')
        
    print(f'All files are correct')
    

### Functions that are used to overwrite built in functions ###

def overwrite_built_in():
    r"""
    This function overwrites some built in functions for issues that cannot be overcome otherwise.
    """
    
    # Overwritten to make sure that when using the imagefolder dataset, that empty folders are still included as classes and don't raise errors
    torchvision.datasets.folder.make_dataset = make_dataset

    # Overwritten to make sure the cache is saved in the __pycache__ folder rather than the run directory for the lsun dataset
    torchvision.datasets.LSUN = LSUN_overwrite 


class LSUN_overwrite(VisionDataset):
    """`LSUN <https://www.yf.io/p/lsun>`_ dataset.
    You will need to install the ``lmdb`` package to use this dataset: run
    ``pip install lmdb``
    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        classes: Union[str, List[str]] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass_overwrite(root=os.path.join(root, f"{c}_lmdb"), transform=transform))  # LSUNClass --> LSUNClass_overwrite

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes: Union[str, List[str]]) -> List[str]:
        categories = [
            "bedroom",
            "bridge",
            "church_outdoor",
            "classroom",
            "conference_room",
            "dining_room",
            "kitchen",
            "living_room",
            "restaurant",
            "tower",
        ]
        dset_opts = ["train", "val", "test"]

        try:
            classes = cast(str, classes)
            verify_str_arg(classes, "classes", dset_opts)
            if classes == "test":
                classes = [classes]
            else:
                classes = [c + "_" + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = "Expected type str or Iterable for argument classes, but got type {}."
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr_type = "Expected type str for elements in argument classes, but got type {}."
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr_type.format(type(c)))
                c_short = c.split("_")
                category, dset_opt = "_".join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class", iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        return "Classes: {classes}".format(**self.__dict__)


class LSUNClass_overwrite(VisionDataset):
    def __init__(
        self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ) -> None:
        import lmdb
        super().__init__(root, transform=transform, target_transform=target_transform)
        

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "__pycache__/_cache_" + "".join(c for c in root if c in string.ascii_letters)  # "_cache_" --> "__pycache__/_cache_"
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.length
    
    def test():
        print("test received")

    
def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    ### This part has been removed s.t. now empty class folders are allowed ###
    # empty_classes = set(class_to_idx.keys()) - available_classes
    # if empty_classes:
    #     msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
    #     if extensions is not None:
    #         msg += f"Supported extensions are: {', '.join(extensions)}"
    #     raise FileNotFoundError(msg)


    return instances
    