import warnings
warnings.filterwarnings("ignore")
import glob
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import itertools
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import functools
import pickle
    
from util import *
    
class _CBINorm(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, num_con=8, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(_CBINorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.ConBias = nn.Sequential(
            nn.Linear(num_con, num_features),
            nn.Tanh()
        )
        
    def _check_input_dim(self, input):
        raise NotImplementedError
        
    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    'Unexpected running stats buffer(s) {names} for {klass} '
                    'with track_running_stats=False. If state_dict is a '
                    'checkpoint saved before 0.4.0, this may be expected '
                    'because {klass} does not track running stats by default '
                    'since 0.4.0. Please remove these keys from state_dict. If '
                    'the running stats are actually needed, instead set '
                    'track_running_stats=True in {klass} to enable them. See '
                    'the documentation of {klass} for details.'
                    .format(names=" and ".join('"{}"'.format(k) for k in running_stats_keys),
                            klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)

        super(_CBINorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
            
    def forward(self, input, ConInfor):
        self._check_input_dim(input)
        b, c = input.size(0), input.size(1)
        tarBias = self.ConBias(ConInfor).view(b,c,1,1)
        out = F.instance_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats, self.momentum, self.eps)
        
        if self.affine:
            bias = self.bias.repeat(b).view(b,c,1,1)
            weight = self.weight.repeat(b).view(b,c,1,1)
            return (out.view(b, c, *input.size()[2:])+tarBias)*weight + bias
        else:
            return out.view(b, c, *input.size()[2:])+tarBias

class CBINorm2d(_CBINorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
            
class _CBBNorm(Module):
    def __init__(self, num_features, num_con, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_CBBNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()
        
        self.ConBias = nn.Sequential(
            nn.Linear(num_con, num_features),
            nn.Tanh()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, ConInfor):
        self._check_input_dim(input)
        b, c = input.size(0), input.size(1)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
                
        out = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        
        biasSor = self.avgpool(out)
        biasTar = self.ConBias(ConInfor).view(b,c,1,1)
        
        if self.affine:
            weight = self.weight.repeat(b).view(b,c,1,1)
            bias = self.bias.repeat(b).view(b,c,1,1)
            return (out - biasSor + biasTar)*weight + bias
        else:
            return out - biasSor + biasTar

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class CBBNorm2d(_CBBNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
            
def get_norm_layer(layer_type='instance', num_con=2):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        c_norm_layer = functools.partial(CBBNorm2d, affine=True, num_con=num_con)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=True, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer
    
#############################################################################################################
############################################# Generator #####################################################
#############################################################################################################

class SingleResidualBlock(nn.Module):
    def __init__(self, nch, c_norm_layer):
        super(SingleResidualBlock, self).__init__()
        self.c1 = nn.Conv2d(nch, nch, kernel_size=3, stride=1, padding=1, bias=False)
        self.cn1 = c_norm_layer(nch)
        self.c2 = nn.Conv2d(nch, nch, kernel_size=3, stride=1, padding=1, bias=False)
        self.cn2 = c_norm_layer(nch)
        
    def forward(self, x):
        data, con = x[0], x[1]
        res = data
        res_out = nn.ReLU()(self.cn1(self.c1(data), con))
        res_out = self.cn2(self.c2(res_out), con)
        return torch.add(res_out, res), con

class SingleGenerator(nn.Module):
    def __init__(self, nch_in, nch, reduce=2, num_cls=3, res_num=6, norm_type="instance", num_con=2, nch_out=None):
        super(SingleGenerator, self).__init__()
        if nch_out==None:
            nch_out = nch_in
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=num_con)
        self.num_cls = num_cls

        # down sampling
        cnn_layers = [nn.Conv2d(nch_in*1, nch, kernel_size=7, stride=1, padding=3, bias=False)]
        cnorms = [c_norm_layer(nch)]
        for i in range(num_cls):
            cnn_layers.append(nn.Conv2d(nch*(2**i), nch*2**(i+1), kernel_size=2*reduce, stride=reduce, padding=int(reduce/2), bias=False))
            cnorms.append(c_norm_layer(nch*2**(i+1)))
        self.down_convs = nn.ModuleList(cnn_layers)
        self.down_cnorms = nn.ModuleList(cnorms)
        
        # residual block
        res_block = []
        for _ in range(res_num):
            res_block.append(SingleResidualBlock(nch*2**(num_cls), c_norm_layer))
        self.resBlocks = nn.Sequential(*res_block)
        
        # up sampling
        cnn_layers = [nn.ConvTranspose2d(nch*(2**(num_cls)), nch*2**(num_cls-1), kernel_size=2*reduce, stride=reduce, padding=int(reduce/2), bias=False)]
        norms = [norm_layer(nch*2**(num_cls-1))]
        for i in range(1, num_cls)[::-1]:
            cnn_layers.append(nn.ConvTranspose2d(nch*(2**(i)), nch*2**(i-1), kernel_size=2*reduce, stride=reduce, padding=int(reduce/2), bias=False))
            norms.append(norm_layer(nch*2**(i-1)))
        cnn_layers.append(nn.Conv2d(nch, nch_out, kernel_size=7, stride=1, padding=3, bias=False))
        self.up_convs = nn.ModuleList(cnn_layers)
        self.up_norms = nn.ModuleList(norms)
        
    def forward(self, x, c):
        for i in range(self.num_cls+1):
            x = self.down_convs[i](x)
            x = self.down_cnorms[i](x, c)
            x = nn.ReLU()(x)
            
        x = self.resBlocks([x, c])[0]
        for i in range(self.num_cls):
            x = self.up_convs[i](x)
            x = self.up_norms[i](x)
            x = nn.ReLU()(x)
        x = self.up_convs[-1](x)
        x = nn.Tanh()(x)
        return x
    
#############################################################################################################
########################################## Discriminator ####################################################
#############################################################################################################

class SingleDiscriminator_original(nn.Module):
    def __init__(self, nch_in, nch, reduce=2, num_cls=3, norm_type="instance", num_con=2):
        super(SingleDiscriminator_original, self).__init__()
        self.num_cls = num_cls

        # down sampling
        
        cnn_layers = [nn.Conv2d(nch_in*1, nch, kernel_size=4, stride=2, padding=1, bias=False),
                     nn.LeakyReLU()]
        
        dim_in = nch
        for i in range(1, num_cls):
            
            dim_out = min(dim_in*2, nch*8)
            cnn_layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=2*reduce, stride=reduce, padding=int(reduce/2), bias=False))
            cnn_layers.append(nn.LeakyReLU())
            dim_in = dim_out
            
        dim_out = min(dim_in*2, nch*8)
        cnn_layers.append(nn.Conv2d(dim_in, 1, kernel_size=4, stride=1, padding=1, bias=True))
        
        self.down_convs = nn.Sequential(*cnn_layers)

    def forward(self, x):
        return self.down_convs(x)
    
class SingleDiscriminator_original_multi(nn.Module):
    
    def __init__(self, nch_in, nch, reduce=2, num_cls=3, norm_type="instance", num_con=2):
        super(SingleDiscriminator_original_multi, self).__init__()
        self.discriminator1 = SingleDiscriminator_original(nch_in, nch, reduce, num_cls, norm_type, num_con)
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.discriminator2 = SingleDiscriminator_original(nch_in, nch//2, reduce, num_cls, norm_type, num_con)
        
    def forward(self, x):
        output1 = self.discriminator1(x)
        output2 = self.discriminator2(self.down(x))
        return [output1, output2]
    
class SingleDiscriminator_solo(nn.Module):
    def __init__(self, nch_in, nch, reduce=2, num_cls=3, norm_type="instance", num_con=2):
        super(SingleDiscriminator_solo, self).__init__()
        
        self.num_cls = num_cls

        # down sampling
        
        cnn_layers = [nn.Conv2d(nch_in*1, nch, kernel_size=4, stride=2, padding=1, bias=False),
                     nn.LeakyReLU()]
        
        dim_in = nch
        for i in range(1, num_cls):
            
            dim_out = min(dim_in*2, nch*8)
            cnn_layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=2*reduce, stride=reduce, padding=int(reduce/2), bias=False))
            cnn_layers.append(nn.LeakyReLU())
            dim_in = dim_out
            
        self.down_convs = nn.Sequential(*cnn_layers)

    def forward(self, x):
        return self.down_convs(x)

class SingleDiscriminator_solo_multi(nn.Module):
    
    def __init__(self, nch_in, nch, reduce=2, num_cls=3, norm_type="instance", n_class=4):
        super(SingleDiscriminator_solo_multi, self).__init__()
        self.n_class = n_class
        self.discriminator1 = SingleDiscriminator_solo(nch_in, nch, reduce, num_cls, norm_type, None)
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.discriminator2 = SingleDiscriminator_solo(nch_in, nch//2, reduce, num_cls, norm_type, None)
        
        dim_in = min(nch*2**num_cls, nch*8)
        self.last_layer1 = nn.Conv2d(dim_in, 1, kernel_size=4, stride=1, padding=1, bias=True)
        self.last_layer2 = nn.Conv2d(dim_in//2, 1, kernel_size=4, stride=1, padding=1, bias=True)
        self.classification_layer1 = [nn.Conv2d(dim_in, n_class, kernel_size=8, stride=1, padding=0, bias=True)]
        self.classification_layer2 = [nn.Conv2d(dim_in//2, n_class, kernel_size=4, stride=1, padding=0, bias=True)]
        
        self.classification_layer1.append(nn.Softmax())
        self.classification_layer2.append(nn.Softmax())
        
        self.classification_layer1 = nn.Sequential(*self.classification_layer1)
        self.classification_layer2 = nn.Sequential(*self.classification_layer2)
        
    def forward(self, x):
        disout1 = self.discriminator1(x)
        disout2 = self.discriminator2(self.down(x))
        output1 = self.last_layer1(disout1)
        output2 = self.last_layer2(disout2)
        out_class1 = self.classification_layer1(disout1)
        out_class2 = self.classification_layer2(disout2)
        return [output1, output2], [out_class1.view(-1, self.n_class), out_class2.view(-1, self.n_class)]

#############################################################################################################
############################################# Encoder #######################################################
#############################################################################################################

class BasicBlock(nn.Module):
    def __init__(self, nch_in, nch_out, c_norm_layer=None):
        super(BasicBlock, self).__init__()
        
        self.cnorm1 = c_norm_layer(nch_in)
        self.nl1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(nch_in, nch_in, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        
        self.cnorm2 = c_norm_layer(nch_in)
        self.nl2 = nn.LeakyReLU(0.2)
        
        self.cmp = nn.Sequential(
            nn.Conv2d(nch_in, nch_out, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect"),
            nn.AvgPool2d(2, 2)
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(nch_in, nch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, input):
        x, d = input
        out = self.cmp(self.nl2(self.cnorm2(self.conv1(self.nl1(self.cnorm1(x,d))),d)))
        out = out + self.shortcut(x)
        return [out,d]

class Encoder_original(nn.Module):
    def __init__(self, nch_in, nch_out, nch=64, num_cls=3, norm_type="instance", num_con=2):
        super(Encoder_original, self).__init__()
        _, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=num_con)
        self.num_cls = num_cls

        self.first_layer = nn.Conv2d(nch_in*1, nch, kernel_size=7, stride=2, padding=1, bias=True)
        
        cnn_layers = []
        in_nch = nch
        for i in range(num_cls):
            out_nch = in_nch * 2
            cnn_layers.append(BasicBlock(in_nch, out_nch, c_norm_layer))
            in_nch = out_nch
        self.layers = nn.Sequential(*cnn_layers)
        self.last_layer = nn.Sequential(nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1))
        self.fcmean = nn.Linear(out_nch, nch_out)
        self.fcvar = nn.Linear(out_nch, nch_out)
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
    def forward(self, x, c):
        x_conv = self.last_layer(self.layers([self.first_layer(x),c])[0])
        b = x_conv.size(0)
        x_conv = x_conv.view(b, -1)
        mu = self.fcmean(x_conv)
        logvar = self.fcvar(x_conv)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
    
class BasicBlock_classification(nn.Module):
    def __init__(self, nch_in, nch_out, norm_layer):
        super(BasicBlock_classification, self).__init__()
        
        self.norm1 = norm_layer(nch_in)
        self.nl1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(nch_in, nch_in, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        
        self.norm2 = norm_layer(nch_in)
        self.nl2 = nn.LeakyReLU(0.2)
        
        self.cmp = nn.Sequential(
            nn.Conv2d(nch_in, nch_out, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect"),
            nn.AvgPool2d(2, 2)
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(nch_in, nch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, input):
        x = input
        out = self.cmp(self.nl2(self.norm2(self.conv1(self.nl1(self.norm1(x))))))
        out = out + self.shortcut(x)
        return out
    
class Encoder(nn.Module):
    def __init__(self, nch_in, nch_out, nch=64, num_cls=3, norm_type="instance", num_con=2):
        super(Encoder, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=num_con)
        self.num_cls = num_cls
        self.first_layer = nn.Conv2d(nch_in*1, nch, kernel_size=7, stride=2, padding=1, bias=True)
        
        cnn_layers = []
        in_nch = nch
        for i in range(num_cls):
            out_nch = in_nch * 2
            cnn_layers.append(BasicBlock_classification(in_nch, out_nch, norm_layer))
            in_nch = out_nch
        self.layers = nn.Sequential(*cnn_layers)
        self.last_layer = nn.Sequential(nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1))
        self.fcmean = nn.Linear(out_nch, nch_out)
        self.fcvar = nn.Linear(out_nch, nch_out)
        self.fcclass = nn.Linear(out_nch, num_con)
        self.fcclass_reduced = nn.Linear(out_nch, num_con)
 
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def freeze_melt(self, classifier_layers, mode="freeze"):
        netE_layers = list(self.state_dict().keys())
        for i, param in enumerate(self.parameters()):
            if netE_layers[i] in classifier_layers:
                if mode=="freeze":
                    param.requires_grad = False
                elif mode=="melt":
                    param.requires_grad = True
        
    def forward(self, x):
        feature = self.last_layer[0](self.layers(self.first_layer(x)))
        x_conv = feature
        mu = self.fcmean(self.last_layer[1](x_conv).view(feature.size(0),-1))
        logvar = self.fcvar(self.last_layer[1](x_conv).view(feature.size(0),-1))
        c_code = self.reparametrize(mu, logvar)
        
        return c_code, mu, logvar, None, None
    
class Positive(torch.autograd.Function):
    def __init__(self):
        super(Positive, self).__init__()
        
    def forward(ctx, input):
        min_ = float(input[input>0].min())
        ctx.save_for_backward(input)
        return input.clamp(min=0, max=min_)/min_
    
    def backward(ctx, dL_dy):
        input, = ctx.saved_variables
        dL_dx = dL_dy.clone()
        dL_dx[input<0] = 0
        return dL_dx
    
class ReLUmodified(torch.autograd.Function):
    def __init__(self):
        super(ReLUmodified, self).__init__()
        
    def forward(ctx, input, a, reference):
        reference = Positive()(reference-a)
        ctx.save_for_backward(input, a, reference)
        return torch.mul(input, reference)
    
    def backward(ctx, dL_dy):
        input, a, reference = ctx.saved_variables
        dL_dx = dL_dy.clone()
        dL_da = dL_dy.clone()
        
        dL_dx[reference==0] = 0
        dL_da[reference==0] = 0
        return dL_dx, -dL_da, torch.zeros(reference.shape, device=reference.device)
    
class GetAttention_relu(nn.Module):
    def __init__(self, requires_grad=False):
        super(GetAttention_relu, self).__init__()
        self.a = nn.Parameter(torch.Tensor([0.]), requires_grad=requires_grad)
        
    def forward(self, input, reference):
        output = ReLUmodified()(input, self.a, reference)
        return output
    
class Encoder_gradattention(nn.Module):
    def __init__(self, nch_in, nch_out, nch=64, num_cls=3, norm_type="instance", num_con=2, attention_mode="relu"):
        super(Encoder_gradattention, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=num_con)
        self.num_cls = num_cls
        self.attention_mode = attention_mode
        if self.attention_mode=="relumodified":
            self.get_attention = GetAttention_relu(True)
        elif self.attention_mode=="relu":
            self.get_attention = GetAttention_relu(False)
        self.first_layer = nn.Conv2d(nch_in*1, nch, kernel_size=7, stride=2, padding=1, bias=True)
        
        cnn_layers = []
        in_nch = nch
        for i in range(num_cls):
            out_nch = in_nch * 2
            cnn_layers.append(BasicBlock_classification(in_nch, out_nch, norm_layer))
            in_nch = out_nch
        self.layers = nn.Sequential(*cnn_layers)
        self.last_layer = nn.Sequential(nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1))
        self.fcmean = nn.Sequential(nn.Linear(out_nch, int(out_nch/2)),
                                    nn.Linear(int(out_nch/2), nch_out))
        self.fcvar = nn.Sequential(nn.Linear(out_nch, int(out_nch/2)),
                                    nn.Linear(int(out_nch/2), nch_out))
        self.fcclass = nn.Linear(out_nch, num_con)
        self.fcclass_reduced = nn.Linear(out_nch, num_con)
 
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def freeze_melt(self, classifier_layers, mode="freeze"):
        netE_layers = list(self.state_dict().keys())
        for i, param in enumerate(self.parameters()):
            if netE_layers[i] in classifier_layers:
                if mode=="freeze":
                    param.requires_grad = False
                elif mode=="melt":
                    param.requires_grad = True
        
    def forward(self, x):
        feature = self.last_layer[0](self.layers(self.first_layer(x)))
        output_class = self.fcclass(self.last_layer[1](feature).view(feature.size(0),-1))
        
        for i in range(len(classes)):
            label = torch.tensor(np.reshape(np.arange(len(classes)), (1, len(classes)))).repeat(x.shape[0], 1)[:,i:i+1]
            loss = criterion_class(output_class, class_encode(label, device, ref_label, "target_only"))
            grads = torch.autograd.grad(loss, feature, retain_graph=True)
            _, _, H, W = grads[0].shape
            w = grads[0].mean(-1).mean(-1).view(x.shape[0], 1024, 1, 1).expand(x.shape[0], 1024, H, W)
            a = feature
            important_feature = torch.mul(a, w)

            if i==0:
                important_features = important_feature
            else:
                important_features = important_features + important_feature
                
        if self.attention_mode == "relu":
            x_conv = self.get_attention(feature, important_features)
            attention = Positive()(x_conv)
        elif self.attention_mode == "relumodified":
            x_conv = self.get_attention(feature, standardize_torch(important_features))
            attention = Positive()(x_conv)
        
        mu = self.fcmean(self.last_layer[1](x_conv).view(feature.size(0),-1))
        logvar = self.fcvar(self.last_layer[1](x_conv).view(feature.size(0),-1))
        c_code = self.reparametrize(mu, logvar)
        
        output_class = self.fcclass_reduced(self.last_layer[1](x_conv).view(feature.size(0),-1))
        output_class = nn.Softmax()(output_class)
            
        return c_code, mu, logvar, output_class, attention
    
class Encoder_classifier(nn.Module):
    def __init__(self, nch_in, nch_out, nch=64, num_cls=3, norm_type="instance", num_con=2):
        super(Encoder_classifier, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=num_con)
        self.num_cls = num_cls

        self.first_layer = nn.Conv2d(nch_in*1, nch, kernel_size=7, stride=2, padding=1, bias=True)
        
        cnn_layers = []
        in_nch = nch
        for i in range(num_cls):
            out_nch = in_nch * 2
            cnn_layers.append(BasicBlock_classification(in_nch, out_nch, norm_layer))
            in_nch = out_nch
        self.layers = nn.Sequential(*cnn_layers)
        self.last_layer = nn.Sequential(nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1))
        
        self.fcclass = nn.Linear(out_nch, num_con)
 
    def forward(self, x):
        x_conv = self.last_layer(self.layers(self.first_layer(x)))
        x_conv = x_conv.view(x_conv.size(0), -1)
        output_class = self.fcclass(x_conv)
        output_class = F.softmax(output_class)
        return output_class