# Code from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from inspect import isfunction

import torch
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
import pysnooper



# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Python starts broadcasting from innermost side, i.e., right most
# for size A[2,6] * B[6],
# B will be expanded as B[1,6]
# Then A will be expanded as A[2,1,6]
# So that the last two dimensions of A could elementwise times B

# import torch

# x = torch.randn(2,  6)  
# print(x)
# emb = torch.randn(6) 
# print(emb)

# x_with_none = x[:, None] # shape [2, 1, 6]  
# print(x_with_none.shape)
# emb_with_none = emb[None, :] # shape [1, 6]
# print(emb_with_none.shape)

# out = x_with_none * emb_with_none

# print(out) 
# print(out.shape)
# # [2, 1, 6]
# out_sqeeze = out.squeeze(-2)
# print(out_sqeeze)
# print(out_sqeeze.shape)

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# norm_nc: Number of channels in the normalized tensor.
# label_nc: Number of channels in the segmentation map, default is 3, so make sure segmap has channels of 3
# eps: A small constant for numerical stability.

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

class SPADEGroupNorm_V1(nn.Module):
    def __init__(self, norm_nc, label_nc, eps = 1e-5):
        super().__init__()

# input = torch.randn(20, 6, 10, 10)
# # Separate 6 channels into 3 groups
# m = nn.GroupNorm(3, 6)
# # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
# m = nn.GroupNorm(6, 6)
# # Put all 6 channels into a single group (equivalent with LayerNorm)
# m = nn.GroupNorm(1, 6)
# # Activating the module
# output = m(input)

        # norm_nc is image channels = 64*2 = 128
        # label_nc is mask channels = 3*2= 6 for conditioned_diffusion, 3 for diffusion

        self.norm = nn.GroupNorm(1, norm_nc, affine=False) # 32/16 # output images have  identical dimensions as inputs

        self.eps = eps
        nhidden = norm_nc # originally 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
    # @pysnooper.snoop()
    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)

        # print(f'====x = {x.shape} ====')
        # print(f'====segmap = {segmap.shape} ====')

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        # print(2)
        actv = self.mlp_shared(segmap)
        # print(3)
        gamma = self.mlp_gamma(actv)
        # print(4)
        beta = self.mlp_beta(actv)
        # print(5)

        # apply scale and bias
        return x * (1 + gamma) + beta

# building block modules

class SPADEGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, eps = 1e-5):
        super().__init__()

# input = torch.randn(20, 6, 10, 10)
# # Separate 6 channels into 3 groups
# m = nn.GroupNorm(3, 6)
# # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
# m = nn.GroupNorm(6, 6)
# # Put all 6 channels into a single group (equivalent with LayerNorm)
# m = nn.GroupNorm(1, 6)
# # Activating the module
# output = m(input)

        # norm_nc is image channels = 64*2 = 128
        # label_nc is mask channels = 3*2= 6 for conditioned_diffusion, 3 for diffusion

        self.norm = nn.GroupNorm(label_nc, label_nc, affine=False) # 32/16 # output images have  identical dimensions as inputs

        self.eps = eps
        nhidden = norm_nc # originally 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
    # @pysnooper.snoop()
    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        # x = self.norm(x)

        # print(f'====x = {x.shape} ====')
        # print(f'====segmap = {segmap.shape} ====')

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        segmap = self.norm(segmap)
        # print(2)
        actv = self.mlp_shared(segmap)
        # print(3)
        gamma = self.mlp_gamma(actv)
        # print(4)
        beta = self.mlp_beta(actv)
        # print(5)

        # apply scale and bias
        return x * (1 + gamma) + beta

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=3, norm=True):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, dim)
        ) if exists(emb_dim) else None
        
        # self.robot_mlp = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(robot_emb_dim, dim)
        # ) if exists(robot_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim) # dim are channel numbers rather than width and height

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
           nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        ) 

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.time_mlp) and exists(time_emb):
            time_condition = self.time_mlp(time_emb)
            h = h + rearrange(time_condition, 'b c -> b c 1 1')

        # if exists(self.robot_mlp) and exists(robot_emb):
        #     robot_condition = self.robot_mlp(robot_emb)
        #     h = h + rearrange(robot_condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)


class ConvNextBlockMaskEmbeddingEncoder(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=3, norm=True):
        super().__init__()

        self.channels = dim
        self.c_channels = 3

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net_before_emb = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            # normalization(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
            # LayerNorm(dim) if norm else nn.Identity(),
            # normalization(dim) if norm else nn.Identity()
        )

        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, dim)  # check if emb_dim == dim?!
        ) if exists(emb_dim) else None

        # self.net_after_emb = nn.Sequential(
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim_out, 3, padding=1)
        # )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # self.channels = dim
        # self.c_channels = 3
        # self.in_norm = SPADEGroupNorm(self.channels, self.c_channels)

    # @pysnooper.snoop()
    def forward(self, x, time_emb=None):

        # print(f'--x = {x.shape}--')

        h = self.ds_conv(x)
        # print(f'--h = {h.shape}--')

        if exists(self.time_mlp) and exists(time_emb):
            # print(f'--time_emb = {time_emb.shape}--')
            time_condition = self.time_mlp(time_emb)
            # print(f'--time_condition = {time_condition.shape}--')

            h = h + rearrange(time_condition, 'b c -> b c 1 1')

        h = self.net_before_emb(h)
        # print(f'----h = {h.shape}----')



            # print(f'------h = {h.shape}------')
        # if exists(self.robot_mlp) and exists(robot_emb):
        #     robot_condition = self.robot_mlp(robot_emb)
        #     h = h + rearrange(robot_condition, 'b c -> b c 1 1')
        # h = self.net_after_emb(h)
        
        return h + self.res_conv(x)

class ConvNextBlockMaskEmbeddingDecoder(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=3, norm=True, mask_channels=None):
        super().__init__()

        self.channels = dim
        self.c_channels = mask_channels # two masks concatenated


        
        # self.robot_mlp = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(robot_emb_dim, dim)
        # ) if exists(robot_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.mask_conv = nn.Conv2d(mask_channels, mask_channels, 7, padding=3, groups=mask_channels)

        # print(f'========norm is {norm}====')
        
        self.spade_norm = SPADEGroupNorm(self.channels, self.c_channels) if norm else nn.Identity()

        # print(f'0000000000000{self.spade_norm}')
        self.net_before_emb = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            # nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
            # SPADEGroupNorm(self.channels, self.c_channels) if norm else nn.Identity()
        )


        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, dim)  # check if emb_dim == dim?! this means embedding is [1,192], and the output will be [1,128]
        ) if exists(emb_dim) else None

        # self.net_after_emb = nn.Sequential(
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim_out, 3, padding=1)  # dim = 128, dim_out = 64
        # )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # self.channels = dim
        # self.c_channels = 3
        # self.in_norm = SPADEGroupNorm(self.channels, self.c_channels)

    # @pysnooper.snoop()
    def forward(self, x, time_emb=None, mask=None):
        # print(f'>>>>>>>>>x={x.shape}<<<<<<<<<')
        h = self.ds_conv(x)
        mask = self.mask_conv(mask)
        
        # print(f'1>>>>>>>>>h={h.shape}<<<<<<<<<')

        # print(f'>>>>>>>>>mask={mask.shape}<<<<<<<<<')
        h = self.spade_norm(h, mask)
        # print(f'2>>>>>>>>>h={h.shape}<<<<<<<<<')
        # print(f'4>>>>>>>>>h={h.shape}<<<<<<<<<')
        if exists(self.time_mlp) and exists(time_emb):
            # print(f'40>>>>>>>>>time_emb={time_emb.shape}<<<<<<<<<')
            time_condition = self.time_mlp(time_emb)
            # print(f'41>>>>>>>>>time_condition={time_condition.shape}<<<<<<<<<')
            h = h + rearrange(time_condition, 'b c -> b c 1 1')        
        h = self.net_before_emb(h)
        # print(f'3>>>>>>>>>h={h.shape}<<<<<<<<<')
        # h = self.spade_norm(h, mask)

        # print(f'5>>>>>>>>>h={h.shape}<<<<<<<<<')
        # if exists(self.robot_mlp) and exists(robot_emb):
        #     robot_condition = self.robot_mlp(robot_emb)
        #     h = h + rearrange(robot_condition, 'b c -> b c 1 1')
        # h = self.net_after_emb(h)
        # print(f'6>>>>>>>>>h={h.shape}<<<<<<<<<')  

        # res_x = self.res_conv(x) 
        # print(f'7>>>>>>>>>res_x={res_x.shape}<<<<<<<<<')  

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)
