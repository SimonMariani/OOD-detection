# OOD-detection


## Dependencies
In order to run the code, the dependencies in the unvertainty.yml file can be installed. However, not all of them are needed for the majority of the functionalities. The main dependencies are enough to run most of the code:

- PyTorch 1.12.0, torchvision 0.13.0, CudaToolkit 11.3
- yaml 0.2.5, pyyaml 6.0
- imread 0.7.4
- matplotlib 3.5.1
- scipy 1.7.3
- pandas 1.4.3

## Running the code
The file experiment.yaml contains the information about the config files directory and the config files to run. All the config files for our experiments are also present and can be altered if needed. Note that it differs per method and setup which parameters are necessary to include in the config file.
This file also contains the paths to the following directories:

- Data directory -> the directory where the datasets are stored
- Run directory -> the directory to which the indivicual runs are stored
- Results directory - > the directory to which the runs are stored, this directory is paralel to the run directory and is therefore overwritten after every experiment with the same name.

The files train.py, inference.py and evaluation.py can be ran indiviudally or all at once by running the run.sh file. All the information is extraced from the experiment.yaml file These are the only runnable files, other files might be necessary for more specific setup changes and/or adding new datasets. 

## Datasets
The datasets can be downloaded and placed in the data directory. Note that some datasets are fairly large and if only used for OOD detection, only the test/val set is needed. Our codebase supports the following datasets:

- CIFAR10*: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR100*: https://www.cs.toronto.edu/~kriz/cifar.html
- SVHN*: http://ufldl.stanford.edu/housenumbers/
- Tiny ImageNet: https://www.image-net.org/index.php
- Tiny ImageNet ODIN: https://github.com/facebookresearch/odin 
- LSUN: https://www.yf.io/p/lsun
- LSUN ODIN: https://github.com/facebookresearch/odin
- SUN: https://groups.csail.mit.edu/vision/SUN/hierarchy.html
- Places: http://places2.csail.mit.edu/
- Textures*: https://www.robots.ox.ac.uk/~vgg/data/dtd/
- iNaturalist: https://www.inaturalist.org/
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- ImageNet-C: https://github.com/hendrycks/robustness
- MNIST*: http://yann.lecun.com/exdb/mnist/
- Fashion-MNIST*: https://github.com/zalandoresearch/fashion-mnist
- Omniglot*: https://github.com/brendenlake/omniglot
- notMNIST: https://www.kaggle.com/datasets/lubaroli/notmnist

\* These datasets do not need to be downloaded and will be downloaded automatically if not present in the data folder already

## Results

![alt text](https://github.com/SimonMariani/OOD-detection/blob/main/images/table.png?raw=true)

