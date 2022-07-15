import torch
import torch.nn as nn
import torchvision.models as torch_models
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision._internally_replaced_utils import load_state_dict_from_url


class ModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def freeze_until(self, freeze_until):
        weight_layers = []
        for child in list(self.children()):
            if hasattr(child, 'weight'):
                weight_layers.append(child)

        # Finally we freeze all the layers except dor the last x layers
        for layer in weight_layers[:freeze_until]:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            print(f'froze layer: {layer}')

    def fix_params(self, show=False):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if show:
                print(f'fixed parameter: {name}')
    
    def unfix_params(self, show=False):
        for name, param in self.named_parameters():
            param.requires_grad = True
            if show:
                print(f'unfixed parameter: {name}')


    def change_input_size(self, in_size, orig_in_size=(3, 32)):

        for key in list(self._modules.keys()):
            if isinstance(self._modules[key], nn.Conv2d):
                break
        
        module = self._modules[key]

        orig_out_numer = (orig_in_size[1] - module.kernel_size[0] + 2 * module.padding[0])
        new_padding = int((orig_out_numer - in_size[1] + module.kernel_size[0]) / 2)
        new_padding = (new_padding, new_padding)

        if isinstance(module.bias, bool):
            bias = False
        else:
            bias = True

        self._modules[key] = nn.Conv2d(in_size[0], module.out_channels, module.kernel_size, module.stride,
                                             new_padding,
                                             module.dilation, module.groups, bias, module.padding_mode)

    def change_output_size(self, out_size):
        
        for key in list(self._modules.keys())[::-1]:
            if isinstance(self._modules[key], nn.Linear):
                break

        module = self._modules[key]

        final_in_size = module.weight.shape[1]
        self._modules[key] = nn.Linear(final_in_size, out_size)


    def get_device(self):
        for val in self._modules.values():
            if hasattr(val, 'weight'):
                return val.weight.device

    
class ResNet(torch_models.resnet.ResNet, ModelWrapper):

    def __init__(self, resnet='resnet34', pretrained=True, progress=False, max_pool=False, small_conv=True, **kwargs):

        if resnet == 'resnet18':
            block, layers, url = BasicBlock, [2, 2, 2, 2], "https://download.pytorch.org/models/resnet18-f37072fd.pth"
        elif resnet == 'resnet34':
            block, layers, url = BasicBlock, [3, 4, 6, 3], "https://download.pytorch.org/models/resnet34-b627a593.pth"
        elif resnet == 'resnet50':
            block, layers, url = Bottleneck, [3, 4, 6, 3], "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        elif resnet == 'resnet101':
            block, layers, url = Bottleneck, [3, 4, 23, 3], "https://download.pytorch.org/models/resnet101-63fe2227.pth"
        elif resnet == 'resnet152':
            block, layers, url = Bottleneck, [3, 8, 36, 3], "https://download.pytorch.org/models/resnet152-394f9c45.pth"
        else:
            raise ValueError(f'resnet {resnet} is not available')
        
        super().__init__(block, layers, **kwargs)
        
        self.max_pool = max_pool

        if pretrained:
            state_dict = load_state_dict_from_url(url, progress=progress)
            self.load_state_dict(state_dict)
            print("loaded pretrained resnet on imagenet")

        if small_conv:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def feature_list(self, x, return_pred=False):
 
        out_list = []

        out = self.relu(self.bn1(self.conv1(x)))

        if self.max_pool:
            out = self.maxpool(out)

        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)

        if not return_pred:
            return out_list

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out_list, out


# Implemented but needs revision
class LeNet(ModelWrapper):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fully1 = nn.Linear(5 * 5 * 50, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fully2 = nn.Linear(500, 10)

        self.feature_list_len = 3

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 50)
        x = self.fully1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fully2(x)
        return x

    def feature_list(self, x, return_pred=False):

        out_list = []
        out = torch.max_pool2d(torch.relu(self.conv1(x)), 2, 2)
        out_list.append(out.view(out.shape[0], out.shape[1], -1).mean(dim=2))

        out = torch.max_pool2d(torch.relu(self.conv2(out)), 2, 2)
        out_list.append(out.view(out.shape[0], out.shape[1], -1).mean(dim=2))

        out = torch.relu(self.fully1(out.view(-1, 5 * 5 * 50)))
        out_list.append(out.view(out.shape[0], out.shape[1], -1).mean(dim=2))

        if not return_pred:
            return out_list
            
        out = self.dropout1(out)
        out = self.fully2(out)

        return out, out_list
