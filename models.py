import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import functools


def param_init(m):  # code adapted from torchvision VGG class
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, NoisyLinear):
        m.reset_param()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type='replicate', norm_layer=nn.InstanceNorm2d, use_dropout=True, use_bias=True):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))

        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetExtractor(nn.Module):
    """ResnetExtractor that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc: int, output_nc: int, shape: tuple, ngf=32, norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            shape               -- the shape of input image e.g.(256,256)
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetExtractor, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            #  当stride不能整除kernel size 时，必须要有output_padding 不然会少一个
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        self.units = output_nc * np.prod(shape)  # this is the number of units for MLP
        for m in self.modules():
            param_init(m)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class AbstractExtractor(nn.Module):
    def __init__(self):
        super(AbstractExtractor, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class SimpleExtractor(AbstractExtractor):
    def __init__(self, obs_shape: tuple, n_frames: int):
        super(SimpleExtractor, self).__init__()
        act = nn.ReLU(inplace=True)
        out_shape = np.array(obs_shape, dtype=int)
        out_shape //= 32
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(96, 160, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(160, 320, kernel_size=3, stride=2, padding=1),
            act,
            nn.Flatten(),
        )
        self.units = 320 * np.prod(out_shape)

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class NoisyLinear(nn.Module):
    """
    NoisyLinear code adapted from
    https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/blob/master/fqf_iqn_qrdqn/network.py
    """

    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self._noise_mode = True

        self.reset_param()
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def reset_param(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def reset_noise(self):
        self.eps_p.copy_(self._f(self.eps_p))
        self.eps_q.copy_(self._f(self.eps_q))

    def noise_mode(self, mode):
        self._noise_mode = mode

    def forward(self, x):
        if self._noise_mode:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


class AbstractFullyConnected(nn.Module):
    def __init__(self, extractor: nn.Module, n_out: int, noisy=False):
        super(AbstractFullyConnected, self).__init__()
        self.noisy = nn.ModuleList()
        self.resetable = nn.ModuleList()
        self.linear_cls = NoisyLinear if noisy else nn.Linear
        self.extractor = extractor
        self.act = nn.ReLU(inplace=True)

    def reset_noise(self):
        for layer in self.noisy:
            layer.reset_noise()

    def noise_mode(self, mode):
        for layer in self.noisy:
            layer.noise_mode(mode)

    def reset_linear(self):
        n = 0
        for layer in self.resetable:
            n += 1
            param_init(layer)
        print(f'{n} linear layers parameter reset successfully')

    def forward(self, x, **kwargs):
        raise NotImplementedError


class SinglePathMLP(AbstractFullyConnected):
    def __init__(self, extractor: nn.Module, n_out: int, noisy=False):
        super(SinglePathMLP, self).__init__(extractor, n_out, noisy)
        self.linear = self.linear_cls(extractor.units, 512)
        self.out = self.linear_cls(512, n_out)
        if noisy:
            self.noisy.append(self.linear)
            self.noisy.append(self.out)

        self.resetable = nn.ModuleList([
            self.linear,
            self.out
        ])

        self.reset_linear()

    def forward(self, x, **kwargs):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.act(x)
        x = self.out(x)
        return x


class DuelingMLP(AbstractFullyConnected):
    def __init__(self, extractor: nn.Module, n_out: int, noisy=False):
        super(DuelingMLP, self).__init__(extractor, n_out, noisy)
        self.linear_val = self.linear_cls(extractor.units, 512)
        self.linear_adv = self.linear_cls(extractor.units, 512)
        self.val = self.linear_cls(512, 1)
        self.adv = self.linear_cls(512, n_out)

        if noisy:
            self.noisy.append(self.linear_val)
            self.noisy.append(self.linear_adv)
            self.noisy.append(self.val)
            self.noisy.append(self.adv)

        self.resetable = nn.ModuleList([
            self.linear_val,
            self.linear_adv,
            self.val,
            self.adv
        ])

        self.reset_linear()

    def forward(self, x, adv_only=False, **kwargs):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        adv = self.linear_adv(x)
        adv = self.act(adv)
        adv = self.adv(adv)
        if adv_only:
            return adv
        val = self.linear_val(x)
        val = self.act(val)
        val = self.val(val)
        x = val + adv - adv.mean(dim=1, keepdim=True)
        return x
