import torch
import torch.nn as nn

import time
import numpy as np



class UncertaintyWrapper(nn.Module):
    def __init__(self, model, num_classes=10, data_sample=None, compress=None, power=10, rand_projec_size=5000, retain_info=None): 
        
        super().__init__()

        self.model = model
        self.num_classes = num_classes
        self.data_sample = data_sample

        self.compress = compress
        self.retain_info = retain_info
        self.power = power
        self.rand_projec_size = rand_projec_size

        self.needs_grad = False
        self.extensions = ['']

        self.model.feature_list = self.model.feature_list if hasattr(self.model, 'feature_list') else self.model.forward

        if compress == 'rp':
            self.calculate_R(data_sample, rand_projec_size)

        self.feature_shapes = self.get_shapes(data_sample)

    def general_uncertainty(self, prob, measure):

        if measure == 'entropy':
            return -1 * torch.sum(prob * torch.log(prob), dim=1)

        if measure == 'max_p':
            max_p = torch.amax(prob, dim=1)
            return -1 * max_p
        
        else:
            raise ValueError(f'the uncertainty measure{measure} is unavailable')

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
    

    def compress_output(self, out, index=0):

        # Averages the pixels per feature map and add a dimension for generalization
        if self.compress == 'mean':
            return out.mean(dim=[2,3]).unsqueeze(1)  

        if self.compress == 'max':
            return out.amax(dim=[2,3]).unsqueeze(1)  

        if self.compress == 'min':
            return out.amin(dim=[2,3]).unsqueeze(1)  

        if self.compress == 'std':
            return out.std(dim=[2,3]).unsqueeze(1) 

        if self.compress == 'quantile':
            return out.flatten(start_dim=2).quantile(q=0.5, dim=2).unsqueeze(1)
    
        if self.compress == 'modes':
            return self.get_modes(out) 

        if self.compress == 'features':
            return out.view(out.shape[0], out.shape[1], -1)  # changes the view to have feature maps and then pixels

        if self.compress == 'pixels':
            return out.view(out.shape[0], out.shape[1], -1).permute(0,2,1)  # changes the view to have pixels and then feature maps
        
        if self.compress == 'rp':
            return (self.R[index] @ out.flatten(start_dim=1).T).T.unsqueeze(1)  # use the random projection matrix for a lower dimensionality

        if self.compress == 'gram':
            return self.get_g_p(out, self.power)

        if self.compress is None:
            return out

    def get_modes(self, out):

        out_new = torch.zeros(out.shape[0], 5, out.shape[1]).to(out.device)
        out = out.flatten(start_dim=2)

        out_new[:,0] = torch.mean(out, dim=2)
        out_new[:,1] = torch.amax(out, dim=2)
        out_new[:,2] = torch.amin(out, dim=2)
        out_new[:,3] = torch.std(out, dim=2)
        out_new[:,4] = torch.quantile(out, q=0.5, dim=2) 

        return out_new

    def get_g_p(self, g_p, power):
  
        powers = torch.arange(1, power+1).to(g_p.device)

        g_p = g_p.reshape(g_p.shape[0], g_p.shape[1], -1).unsqueeze(1)
        g_p = g_p.repeat(1,power,1,1)**powers.reshape(1,power,1,1)
        
        g_p = torch.einsum('ijfp,ijkp -> ijfk', g_p, g_p)  # (bs, po, fm, p) -> (bs, po, fm, fm)
        g_p = torch.sum(g_p, dim=3)   # (bs, po, fm, fm) -> (bs, po, fm)
        g_p = torch.sign(g_p) * torch.abs(g_p)**(1/powers.reshape(1,power,1))

        return g_p


    def get_shapes(self, sample):
        
        features = self.model.feature_list(sample)

        feature_shapes = []
        for i, feature in enumerate(features):
            feature_shapes.append(self.compress_output(feature, i)[0].shape)

        return feature_shapes

    def calculate_R(self, data_sample, rand_projec_size):

        self.compress = None
        feature_shapes = self.get_shapes(data_sample)

        R = []
        for feature_size in feature_shapes:
            
            num_features = np.prod(list(feature_size))
            num_samples = rand_projec_size*num_features
            samples =  torch.multinomial(torch.tensor([2/3, 1/6, 1/6]), num_samples, replacement=True)
            r = samples.reshape((rand_projec_size, num_features))
            r[r==2] = -1
            r = r * torch.sqrt(torch.tensor([3]))
            R.append(r)
        
        self.R = torch.nn.ParameterList([torch.nn.Parameter(r, requires_grad=False) for r in R])
        self.compress = 'rp'


    def update(self, x=None, y=None):
        r"""Required empty function that every method should have and that can be overwritten if necessary.
        Generally is is used to update aspects during training that are not part of the regular training."""
        pass

    def post_proces(self, x=None, y=None):
        r"""Required empty function that every method should have and that can be overwritten if necessary.
        Generally is is used to update aspects during training that are not part of the regular training."""
        pass



class BasicModel(UncertaintyWrapper):

    def __init__(self, model, num_classes=10, data_sample=None, unc_measure='max_p', prob='softmax', **kwargs):
        super().__init__(model=model, num_classes=num_classes, data_sample=data_sample, **kwargs)

        self.unc_measure = unc_measure
        self.prob = prob
        
    def forward(self, x):
        x = self.model(x)
        return x

    def uncertainty(self, x):
        logits = self.forward(x)
        
        if self.prob == 'softmax':
            prob = torch.softmax(logits, dim=1)

        elif self.prob == 'logits':
            prob = logits

        elif self.prob == 'sigmoid':
            prob = torch.sigmoid(((logits+4) / 16) - 5)

        else:
            raise ValueError(f'probability {self.prob} is not available')

        return self.general_uncertainty(prob, measure=self.unc_measure)

 
class Mahalanobis(UncertaintyWrapper):

    def __init__(self, model, num_classes=10, data_sample=None, cov_type='tied', num_batches=None, print_every=500, 
                use_layers=(0,None), retain_info=None, unc_measure='maha_mean', compress='mean', **kwargs):
        super().__init__(model=model, num_classes=num_classes, data_sample=data_sample, compress=compress, retain_info=retain_info, **kwargs)

        # Moment settings
        self.cov_type = cov_type
        self.num_batches = num_batches
        self.print_every = print_every
        
        # Reduction settings
        self.unc_measure = unc_measure
        self.retain_info = retain_info

        # Layer settings
        self.use_layers = use_layers
        self.equal_features = True if self.feature_shapes[0][0] == self.feature_shapes[len(self.feature_shapes)-1][0] else False

        self.start = self.use_layers[0] if self.use_layers[0] >= 0 else len(self.feature_shapes) + self.use_layers[0]
        self.end = self.use_layers[1] if self.use_layers[1] is not None else len(self.feature_shapes)

        if self.retain_info == 'layers':
            self.extensions = [f'l{layer}' for layer in range(self.start, self.end)]
        elif self.retain_info == 'features' and self.compress == 'modes':
            self.extensions = ['mean', 'max', 'min', 'std', 'quantile']
        
        # Initialize moments
        self.reset_moments()

    def forward(self, x):

        # Obtain the features per layer from the model and calculate the distance per layer
        features = self.model.feature_list(x)

        mahalanobis_distance = []
        for i in range(self.start, self.end):
            gaussian_scores = self.compute_dist_maha(self.compress_output(features[i], i), self.mean_layer_class[i], self.precision_layer[i]) # (bs, x, nc)

            if not self.equal_features:  # if the feature shapes are not equal we need to reduce them already (only occurs with pixels and features)
                gaussian_scores = torch.mean(gaussian_scores, dim=1, keepdim=True)

            mahalanobis_distance.append(gaussian_scores.unsqueeze(3))  # (bs, x nc, 1)  (the 1 is the layer)
        
        mahalanobis_distance = torch.cat(mahalanobis_distance, dim=3)  # -> (bs, x, nc, nl)
        mahalanobis_distance = mahalanobis_distance.permute(0,2,1,3)  # -> (bs, nc, x, nl)

        # We can now choose what information we want to keep from the distribution
        if self.retain_info == 'full':  # full means to return all the information without reduction
            return mahalanobis_distance
        elif self.retain_info == 'layers':  # layers means to retain the layer information
            reduce_dims = [2]
        elif self.retain_info == 'features':  # modes means to maintain the feature information
            reduce_dims = [3]
        else:
            reduce_dims = [2,3]

        # We then reduce the information to 
        if self.unc_measure == 'maha_mean':
            mahalanobis_distance = torch.mean(mahalanobis_distance, dim=reduce_dims)  # -> (bs, nc, x), (bs, nc, nl), (bs, nc) 

        elif self.unc_measure == 'maha_sum':
            mahalanobis_distance = torch.sum(mahalanobis_distance, dim=reduce_dims)  # -> (bs, nc, x), (bs, nc, nl), (bs, nc) 
            
        elif self.unc_measure == 'maha_max':
            mahalanobis_distance = torch.amax(mahalanobis_distance, dim=reduce_dims)  # -> (bs, nc, x), (bs, nc, nl), (bs, nc) 

        elif self.unc_measure == 'maha_min':
            mahalanobis_distance = torch.amin(mahalanobis_distance, dim=reduce_dims)  # -> (bs, nc, x), (bs, nc, nl), (bs, nc) 
            
        return mahalanobis_distance

    def uncertainty(self, x):

        mahalanobis_distance = self.forward(x)  # (bs, x, nc, nl) or (bs, nc)

        if self.retain_info == 'full':  # (bs, x, nc, nl)
            return mahalanobis_distance  

        mahalanobis_distance = self.general_uncertainty(mahalanobis_distance, measure='max_p')  # -> (bs), (bs, nl), (bs, x)

        return mahalanobis_distance

    def compute_dist_maha(self, samples, mean, precision): 

        # samples -> (bs, f_dim, pixels), mean -> (n_classes, f_dim, pixels), precision -> (f_dim, pixels, pixles) or precision -> (nc, f_dim, pixels, pixels)
        zero_f = samples.unsqueeze(0) - mean.unsqueeze(1)  # (1, bs, fdim, pixels) - (nc, 1, fdim, pixels) -> (nc, bs, fdim, pixels)

        if self.cov_type == 'full':
            # (nc, bs, fdim, pixels) x (nc, fdim, pixels, pixels) -> (nc, bs, fdim, pixels)
            term_gau = torch.einsum('cifp, cfpk -> cifk', zero_f, precision) 

        elif self.cov_type == 'tied':
            # (nc, bs, fdim, pixels) x (fdim, pixels, pixels) -> (nc, bs, fdim, pixels)
            term_gau = torch.einsum('cifp, fpk -> cifk', zero_f, precision) 

        term_gau = -0.5 * torch.einsum('cifp, cjfp -> cfij', term_gau, zero_f) #(nc, bs, fdim, pixels) x (nc, bs, fdim, pixels) -> (nc, fdim, bs, bs)
        term_gau = torch.diagonal(term_gau, dim1=2, dim2=3).permute(2,1,0)  # (nc, fdim, bs, bs) -> (bs, fdim, nc)
    
        return term_gau 
    
    
    def mean_precision(self, dataloader, device):

        self.reset_moments()

        for moment in ['mean', 'covariance']: 
            
            print(f'Calculating {moment}')

            num_samples = [[0 for j in range(self.num_classes)] for i in range(len(self.feature_shapes))]    

            start_time = time.time()

            for iteration, (img, targets) in enumerate(dataloader):

                img, targets = img.to(device), targets.to(device)
                features = self.model.feature_list(img)

                batch_indexes = torch.arange(0, len(targets)).to(device)

                for i in range(len(features)):
                    for j in range(self.num_classes):
                        
                        if len(targets.shape) > 1:   # We assume that if the targets contain multiple entries that they are one hot encoded
                            targets = torch.argmax(targets, dim=1)
                            
                        class_indexes = batch_indexes[(targets == j)] 
                        class_features = torch.index_select(self.compress_output(features[i], i), dim=0, index=class_indexes)

                        if class_features.shape[0] == 0:
                            continue

                        num_samples[i][j] += len(class_features)

                        if moment == 'mean':
                            self.mean_layer_class[i][j] += torch.sum((class_features - self.mean_layer_class[i][j]) \
                                                                     / num_samples[i][j], dim=0)

                        elif moment == 'covariance':

                            diff = class_features - self.mean_layer_class[i][j]

                            if self.cov_type == 'full':
                                self.precision_layer[i][j] += ( torch.einsum('ifp,ifk -> fpk', diff, diff) / num_samples[i][j] - \
                                                ((len(diff) / num_samples[i][j]) * self.precision_layer[i][j]) )

                            elif self.cov_type == 'tied':
                                self.precision_layer[i] += ( torch.einsum('ifp,ifk -> fpk', diff, diff) / sum(num_samples[i]) - \
                                                ((len(diff) / sum(num_samples[i])) * self.precision_layer[i]) )

                if self.print_every is not None:
                    if (iteration + 1) % self.print_every == 0:
                        total_time = time.time() - start_time
                        print(f'processed {str(iteration + 1):5s} batches/{str(sum(num_samples[0])):8s} samples,',
                              f'total runtime: {str(np.round(total_time, 3)):5s} seconds,',
                              f'{str(np.round(total_time / 60, 3)):5s} minutes, {str(np.round(total_time / 3600, 3)):5s} hours')
        
                if self.num_batches is not None:
                    if iteration + 1 >= self.num_batches:
                        break

        self.covariance = [cov for cov in self.precision_layer]

        if self.cov_type == 'full':
            for i in range(len(features)):
                for j in range(self.num_classes):
                    for k in range(len(self.precision_layer[i][j])):
                        self.precision_layer[i][j][k] = torch.nn.Parameter(torch.linalg.pinv(self.precision_layer[i][j][k]), requires_grad=self.precision_layer[i][j][k].requires_grad)
        
        elif self.cov_type == 'tied':
            for i in range(len(features)):
                for j in range(len(self.precision_layer[i])):
                    self.precision_layer[i][j] = torch.nn.Parameter(torch.linalg.pinv(self.precision_layer[i][j]), requires_grad=self.precision_layer[i][j].requires_grad)
        
        self.precision_layer = torch.nn.ParameterList(self.precision_layer)

        return self.mean_layer_class, self.precision_layer

    def reset_moments(self):
        
        device = self.model.get_device()
   
        mean_list = [torch.zeros(([self.num_classes] + list(shape))).to(device) for shape in self.feature_shapes]
    
        if self.cov_type == 'full':
            prec_list = [torch.zeros(([self.num_classes] + list(shape) + [shape[-1]])).to(device) for shape in self.feature_shapes]
        elif self.cov_type == 'tied':
            prec_list = [torch.zeros((list(shape) + [shape[-1]])).to(device) for shape in self.feature_shapes]
        else: 
            raise ValueError(f'covariance type {self.cov_type} unavailable')
        
        self.mean_layer_class = torch.nn.ParameterList([torch.nn.Parameter(val, requires_grad=True) for val in mean_list])
        self.precision_layer = torch.nn.ParameterList( [torch.nn.Parameter(val, requires_grad=True) for val in prec_list])        
    
    def post_proces(self, dataloader, device):
        
        self.to(device)
        self.eval()
        with torch.no_grad():
            self.mean_precision(dataloader, device)
        self.train()


class Gram(UncertaintyWrapper):

    def __init__(self, model, num_classes=10, data_sample=None, use_layers=(0,None), retain_info=None, compress='gram', power=10, 
                num_batches=None, print_every=50, unc_measure='gram_sum', **kwargs):
        super().__init__(model=model, num_classes=num_classes, data_sample=data_sample, compress=compress, 
                         power=power, **kwargs)

        # Layer settings
        self.use_layers = use_layers
        self.unc_measure = unc_measure
        self.retain_info = retain_info

        self.num_batches = num_batches
        self.print_every = print_every

        self.equal_features = True if self.feature_shapes[0][0] == self.feature_shapes[len(self.feature_shapes)-1][0] else False

        self.start = self.use_layers[0] if self.use_layers[0] >= 0 else len(self.feature_shapes) + self.use_layers[0]
        self.end = self.use_layers[1] if self.use_layers[1] is not None else len(self.feature_shapes)

        if self.retain_info == 'layers':
            self.extensions = [f'l{layer}' for layer in range(self.start, self.end)]
        elif self.retain_info == 'features' and self.compress == 'modes':
            self.extensions = ['mean', 'max', 'min', 'std', 'quantile']
            
        self.reset_moments()

    def forward(self, x, return_logits=False):  

        features, logits = self.model.feature_list(x, return_pred=True)

        deviations = []
        for i in range(self.start, self.end):  # every layer
           dev = self.get_deviations(self.compress_output(features[i], i), self.min_list[i], self.max_list[i])  # (bs, p, f, c)
           dev = torch.sum(dev, dim=2) # sum over the feature maps -> (bs, p, c)

           if not self.equal_features:  # if the feature shapes are not equal we need to reduce them already (only occurs with pixels and features)
                dev = torch.sum(dev, dim=1, keepdim=True)
                
           deviations.append(dev)  
        deviations = torch.stack(deviations, dim=3) # -> (bs, p, c, l)
        deviations = deviations.permute(0,2,1,3)  # -> (bs, c, p, l)

        # We can now choose what information we want to keep from the distribution
        if self.retain_info == 'full':  # full means to return all the information without reduction
            return deviations
        elif self.retain_info == 'layers':  # layers means to retain the layer information
            reduce_dims = [2]
        elif self.retain_info == 'features':  # modes means to maintain the feature information
            reduce_dims = [3]
        else:
            reduce_dims = [2,3]

        # We then reduce the information to 
        if self.unc_measure == 'gram_mean':
            deviations = torch.mean(deviations, dim=reduce_dims)  # -> (bs, nc, x), (bs, nc, nl), (bs, nc) 

        elif self.unc_measure == 'gram_sum':
            deviations = torch.sum(deviations, dim=reduce_dims)  # -> (bs, nc, x), (bs, nc, nl), (bs, nc) 
            
        elif self.unc_measure == 'gram_max':
            deviations = torch.amax(deviations, dim=reduce_dims)  # -> (bs, nc, x), (bs, nc, nl), (bs, nc) 

        elif self.unc_measure == 'gram_min':
            deviations = torch.amin(deviations, dim=reduce_dims)  # -> (bs, nc, x), (bs, nc, nl), (bs, nc) 
        
        if return_logits:
            return -1*deviations, logits

        return -1*deviations

    def uncertainty(self, x):
        
        deviations, logits = self.forward(x, return_logits=True)
        pred = torch.argmax(logits, dim=1, keepdim=True)

        pred = pred.unsqueeze(2).repeat(1,1,deviations.shape[2]) if len(deviations.shape) == 3 else pred

        uncertainty = -1 * torch.gather(deviations, dim=1, index=pred)
        uncertainty = uncertainty.squeeze(1)
       
        return uncertainty

    def get_deviations(self, x, min, max):

        min_dev = torch.relu(min.unsqueeze(0) - x.unsqueeze(1)) / torch.abs(min.unsqueeze(0)+1e-6)
        max_dev = torch.relu(x.unsqueeze(1) - max.unsqueeze(0)) / torch.abs(max.unsqueeze(0)+1e-6)
        dev = (min_dev + max_dev).permute(0,2,3,1)  # (bs, nc, p, f) -> (bs, p, f, nc)

        return dev


    def min_max(self, dataloader, device):
        
        start_time = time.time()

        for iteration, (img, _) in enumerate(dataloader):

            img = img.to(device)
            features, logits, = self.model.feature_list(img, return_pred=True)

            pred = torch.argmax(logits, dim=1)
            batch_indexes = torch.arange(0, len(logits)).to(device)

            for i in range(self.start, self.end):  # every layer
                for j in range(self.num_classes):  # every class
                    
                    class_indexes = batch_indexes[(pred == j)] 

                    if class_indexes.shape[0] == 0:
                        continue

                    class_features = torch.index_select(self.compress_output(features[i], i), dim=0, index=class_indexes)  # select only the current classes

                    self.min_list[i][j] = torch.amin(torch.concat([self.min_list[i][j].unsqueeze(0), class_features]), dim=0)  # (l, c, p, f)
                    self.max_list[i][j] = torch.amax(torch.concat([self.max_list[i][j].unsqueeze(0), class_features]), dim=0)

            if self.print_every is not None:
                if (iteration + 1) % self.print_every == 0:
                    total_time = time.time() - start_time
                    print(f'processed {str(iteration + 1):5s} batches/{str(iteration*dataloader.batch_size):8s} samples,',
                            f'total runtime: {str(np.round(total_time, 3)):5s} seconds,',
                            f'{str(np.round(total_time / 60, 3)):5s} minutes, {str(np.round(total_time / 3600, 3)):5s} hours')
        
            if self.num_batches is not None:
                if iteration + 1 >= self.num_batches:
                    break
        
    def reset_moments(self):
        
        device = self.model.get_device()
        
        min_list = [torch.ones(([self.num_classes] + list(shape))).to(device) * float('inf') for shape in self.feature_shapes]  
        max_list = [torch.ones(([self.num_classes] + list(shape))).to(device) * float('-inf')for shape in self.feature_shapes]
        
        self.min_list = torch.nn.ParameterList([torch.nn.Parameter(val, requires_grad=True) for val in min_list])
        self.max_list = torch.nn.ParameterList([torch.nn.Parameter(val, requires_grad=True) for val in max_list])
    
    def post_proces(self, dataloader, device):
        
        self.to(device)
        self.eval()
        with torch.no_grad():
            self.min_max(dataloader, device)
        self.train()




# Implemented methods but require revision 

class DUQ(UncertaintyWrapper):  
    
    def __init__(self, model, num_classes=10, data_sample=None, embedding_size=20, model_out_size=10, length_scale=0.1,
                 learn_length_scale=False, gamma=0.999, **kwargs):
        super().__init__(model=model, num_classes=num_classes, data_sample=data_sample, **kwargs)
        
        # TODO make sure everythin corresponds to what the authors did and possibly extend by using the penultimate layers instad of the logits

        # Method settings
        self.W = nn.Parameter(torch.zeros(embedding_size, num_classes, model_out_size))
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer("m", torch.normal(torch.zeros(embedding_size, num_classes), 0.05))
        self.m = self.m * self.N

        if learn_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

        self.gamma = gamma
        self.embedding_size = embedding_size
        self.model_out_size = model_out_size
        self.length_scale = length_scale
        self.learn_length_scale = learn_length_scale
        
        # Additional settings
        self.feature_shapes = self.get_shapes(data_sample)

        self.unc_measure = 'duq'

    def forward(self, x):
        x = self.model(x)
        z = torch.einsum("ij,mnj->imn", x, self.W)
        centroids = self.m / self.N.unsqueeze(0)
        diff = z - centroids.unsqueeze(0)
        distance = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()
        return distance

    def uncertainty(self, x):
        distance = self.forward(x)
        kernel_distance, _ = torch.max(distance, dim=1)

        if self.unc_measure == 'duq':
            return -1 * kernel_distance
        
        prob = torch.softmax(kernel_distance, dim=1)
        return self.regular_uncertainty(prob, measure=self.unc_measure)
    
    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * torch.sum(y, dim=0)
        logits = self.model(x)
        z = torch.einsum("ij,mnj->imn", logits, self.W)
        features_sum = torch.einsum("ijk,ik->jk", z, y)
        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    def update(self, x, y):
        self.eval()
        with torch.no_grad():
            self.update_embeddings(x, y)


class ODIN(UncertaintyWrapper):

    def __init__(self, model, num_classes=10, data_sample=None, temperature=1000,
                criterion=None, magnitude=0.1, data_std=(0.2023, 0.1994, 0.201), **kwargs):
        super().__init__(model=model, num_classes=num_classes, data_sample=data_sample, **kwargs)
        
        # TODO double check if everything here corresponds to what the authors did

        if self.data_sample is not None:
            self.feature_shapes = self.get_shapes(data_sample)
            
        self.needs_grad = True
        self.unc_measure = 'max_p'

        self.temperature = temperature
        self.criterion = criterion
        self.magnitude = magnitude
        self.data_std = data_std
        
    def forward(self, x):
        logits = self.model(x)
        return logits

    def uncertainty(self, x):
        x.requires_grad_(True)

        logits = self.model(x)
        logits = logits / self.temperature

        pred = torch.argmax(logits, dim=1)

        loss = self.criterion(logits, pred)
        loss.backward()
        
        gradient = torch.ge(x.grad, 0)
        gradient = gradient.float() * 2 - 1
        
        gradient[:,0] = (gradient[:,0])/self.data_std[0]
        gradient[:,1] = (gradient[:,1])/self.data_std[1]
        gradient[:,2] = (gradient[:,2])/self.data_std[2]
        
        x_temp = x - self.magnitude * gradient 
        #x_temp = torch.add(x, -self.magnitude, gradient) # should be the same but is slightly different (same within tolerance)

        logits = self.model(x_temp)
        logits = logits / self.temperature
        
        prob = torch.softmax(logits, dim=1)
        
        return self.general_uncertainty(prob, measure=self.unc_measure)
    

class DPN(UncertaintyWrapper):

    def __init__(self, model, num_classes=10, data_sample=None, **kwargs):
        super().__init__(model=model, num_classes=num_classes, data_sample=data_sample, **kwargs)
        
        # TODO make sure everything corresponds to what the authors did

        self.feature_shapes = self.get_shapes(data_sample)

        self.unc_measure = 'max_p'
        
    def forward(self, x):
        x = self.model(x)
        alphas = torch.exp(x)
        return alphas

    def uncertainty(self, x):
        alpha = self.forward(x)
        precision = torch.sum(alpha, dim=1, keepdims=True)
        prob = alpha / precision
        return self.general_uncertainty(prob, measure=self.unc_measure)


class EDL(UncertaintyWrapper):

    def __init__(self, model, num_classes=10, data_sample=None, **kwargs):
        super().__init__(model=model, num_classes=num_classes, data_sample=data_sample, **kwargs)
        
        # TODO make sure everything correspods to what the authors did

        self.feature_shapes = self.get_shapes(data_sample)

        self.unc_measure = 'edl'
        
    def forward(self, x):
        x = self.model(x)
        evidence = torch.relu(x)
        alpha = evidence + 1
        return alpha

    def uncertainty(self, x):

        alpha = self.forward(x)
        precision = torch.sum(alpha, dim=1, keepdims=True)
        prob = alpha / precision

        if self.unc_measure == 'edl':
            return (alpha.shape[1] / precision).squeeze(1)
        
        return self.general_uncertainty(prob, measure=self.unc_measure)



# Not fully implemented method

class Ensemble(UncertaintyWrapper):
    def __init__(self, model, num_classes=10, data_sample=None, unc_measure='split', **kwargs):
        super().__init__(model, num_classes, data_sample, **kwargs)
        
        # This method is not ready yes
        raise NotImplementedError

        self.unc_measure = unc_measure
        self.model = torch.nn.ModuleList(self.model)
        
    def forward(self, x):
        
        outputs = []
        for sub_model in self.model:
            outputs.append(sub_model(x))
        outputs = torch.stack(outputs, dim=-1)

        return torch.mean(outputs, dim=-1) 

    def uncertainty(self, x):
        
        #print("here")
        outputs = []
        # print("here")
        for sub_model in self.model:
            unc = sub_model.uncertainty(x)
            # print(unc.unique())
            outputs.append(unc)
        outputs = torch.stack(outputs, dim=-1)

        if self.unc_measure == 'mean':
            return torch.mean(outputs, dim=-1) 

        if self.unc_measure == 'split':
            mask = outputs[:,1] < 0.005
            out = torch.where(mask, outputs[:,0], outputs[:,1])

            print(outputs[:,1].unique())
            # print(mask.unique())
            # print(out.unique())

            return out

