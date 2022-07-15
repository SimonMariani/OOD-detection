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

\begin{table*}
    \centering
\begin{tabular}{llccc}
\hline
     &          & \multicolumn{3}{c}{MSP / Mahalanobis / MSP + erase / Mahalanobis + erase} \\
  ID &  OOD &                                                 AUROC &                      Detection acc. &                      TNR at TPR 95\% \\
\hline
CIFAR10 & SVHN &                    .947 / .966 / .971 / \textbf{.987} &  .920 / .941 / .950 / \textbf{.958} &  .662 / .786 / .830 / \textbf{.938} \\
     & TinyImageNet &                    .877 / .895 / \textbf{.897} / .891 &  .827 / .829 / \textbf{.845} / .824 &  .461 / .498 / \textbf{.533} / .456 \\
     & LSUN &                    .911 / .913 / \textbf{.930} / .911 &  .857 / .850 / \textbf{.878} / .848 &  .511 / .514 / \textbf{.595} / .475 \\
     & Places &                    .893 / .899 / \textbf{.912} / .899 &  .872 / .878 / \textbf{.889} / .884 &  .494 / .500 / \textbf{.563} / .470 \\
     & CIFAR100 &                    .881 / .896 / \textbf{.905} / .895 &  .827 / .830 / \textbf{.852} / .826 &  .446 / .492 / \textbf{.525} / .457 \\
     & Textures &                    .914 / .968 / .928 / \textbf{.983} &  .855 / .905 / .873 / \textbf{.933} &  .544 / .799 / .594 / \textbf{.903} \\
\hline
CIFAR100 & SVHN &                    .723 / .840 / .785 / \textbf{.894} &  .777 / .817 / .811 / \textbf{.869} &  .142 / .363 / .197 / \textbf{.412} \\
     & TinyImageNet &                    .801 / \textbf{.806} / .790 / .737 &  .737 / \textbf{.745} / .728 / .687 &  \textbf{.246} / .236 / .227 / .119 \\
     & LSUN &                    \textbf{.749} / .726 / .748 / .645 &  \textbf{.700} / .685 / .697 / .624 &  \textbf{.152} / .122 / .141 / .054 \\
     & Places &                    \textbf{.775} / .766 / .773 / .699 &  .818 / .805 / \textbf{.826} / .802 &  \textbf{.207} / .201 / .195 / .094 \\
     & CIFAR10 &                    \textbf{.783} / .753 / .775 / .628 &  \textbf{.719} / .708 / .713 / .612 &  \textbf{.216} / .157 / .197 / .042 \\
     & Textures &                    .787 / .931 / .805 / \textbf{.951} &  .718 / .852 / .727 / \textbf{.882} &  .204 / .658 / .248 / \textbf{.745} \\
\hline
SVHN & CIFAR10 &                    .913 / .983 / .930 / \textbf{.994} &  .893 / .942 / .891 / \textbf{.967} &  .715 / .922 / .721 / \textbf{.982} \\
     & TinyImageNet &                    .915 / .984 / .923 / \textbf{.995} &  .895 / .944 / .889 / \textbf{.970} &  .725 / .927 / .714 / \textbf{.983} \\
     & LSUN &                    .899 / .981 / .906 / \textbf{.994} &  .885 / .939 / .879 / \textbf{.968} &  .680 / .906 / .674 / \textbf{.985} \\
     & Places &                    .909 / .984 / .920 / \textbf{.995} &  .869 / .950 / .867 / \textbf{.973} &  .704 / .921 / .703 / \textbf{.988} \\
     & CIFAR100 &                    .913 / .983 / .923 / \textbf{.993} &  .892 / .941 / .888 / \textbf{.965} &  .713 / .917 / .711 / \textbf{.980} \\
     & Textures &                    .893 / .991 / .890 / \textbf{.997} &  .904 / .963 / .898 / \textbf{.980} &  .684 / .963 / .651 / \textbf{.991} \\
\hline
\end{tabular}
    \caption{The OOD detection results of the MSP and Mahalanobis distance on the default setup, as well as the MSP and Mahalanobis distance on the setup with the erase data augmentation.}
    \label{performances}
\end{table*}

