import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pdb


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],use_se=True):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, num_groups=4, is_rgb_input=False):
        """Initialize the Non-Local Block for red-green color vision deficiency enhancement.

        This module implements a non-local attention mechanism to capture long-range dependencies
        in images, with red-green channel guidance and multi-scale fusion. It is designed to enhance
        images for red-green color vision deficiency (CVD) by focusing on color and structural features.

        Parameters:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            num_groups (int, optional): Number of groups for grouped attention computation (default: 4).
            is_rgb_input (bool, optional): If True, input is RGB; otherwise, a projection layer is used
                                          to map to RGB space (default: False).

        Attributes:
            query_conv (nn.Conv2d): 1x1 convolution to generate query (Q) features.
            key_conv (nn.Conv2d): 1x1 convolution to generate key (K) features.
            value_conv (nn.Conv2d): 1x1 convolution to generate value (V) features.
            pool (nn.AvgPool2d): Average pooling for multi-scale fusion.
            gamma (nn.Parameter): Learnable scaling factor for attention output.
            softmax (nn.Softmax): Softmax layer for attention weights.
            scale_weight1 (nn.Parameter): Scaling factor for attention output.
            scale_weight2 (nn.Parameter): Scaling factor for pooled attention output.
            smooth_conv (nn.Conv2d): 3x3 convolution for smoothing pooled features.
            rg_project (nn.Conv2d, optional): 1x1 convolution to project non-RGB inputs to RGB space.

        Notes:
            - The red-green channel guidance enhances attention to color differences critical for CVD.
            - Multi-scale fusion uses average pooling to capture features at different resolutions.
            - The module is inspired by Wang et al., "Non-Local Neural Networks," CVPR 2018.
        """
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups  # 分组数，可调参数
        self.is_rgb_input = is_rgb_input  # 保存为实例属性
         # 1x1 convolutions for query, key, and value
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
         # Average pooling for multi-scale fusion
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Learnable scaling factor
        self.gamma = nn.Parameter(torch.zeros(1))

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

        # Scaling weights for attention and pooled outputs
        self.scale_weight1 = nn.Parameter(torch.ones(1))
        self.scale_weight2 = nn.Parameter(torch.ones(1))

        # 3x3 convolution for smoothing pooled features
        self.smooth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Projection layer for non-RGB inputs
        if not is_rgb_input:
            self.rg_project = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        """Apply non-local attention with red-green channel guidance and multi-scale fusion.

        This function computes non-local attention on the input tensor, using red-green channel
        differences to guide attention weights. It supports grouped attention for large feature maps
        and multi-scale fusion via average pooling to enhance feature representation for CVD enhancement.

        Parameters:
            x (torch.Tensor): Input tensor, shape [B, C, H, W], values typically in [-1, 1].

        Returns:
            torch.Tensor: Output tensor with same shape as input, enhanced with non-local features.

        Notes:
            - Red-green channel guidance is computed using the absolute difference between red and green channels.
            - Grouped attention reduces memory usage for large feature maps (H*W > num_groups).
            - Multi-scale fusion uses average pooling and bilinear upsampling to combine features at different scales.
        """
        batch_size, C, H, W = x.size()
        
        # Red-green channel guidance
        if self.is_rgb_input and C == 3:
            rg_diff = torch.abs(x[:, 0:1, :, :] - x[:, 1:2, :, :])
            rg_weight = torch.sigmoid(rg_diff)
        else:
            rgb_proj = self.rg_project(x)
            rg_diff = torch.abs(rgb_proj[:, 0:1, :, :] - rgb_proj[:, 1:2, :, :])
            rg_weight = torch.sigmoid(rg_diff)
        
        # Generate query, key, and value
        query = self.query_conv(x * rg_weight)
        key = self.key_conv(x * rg_weight)
        value = self.value_conv(x)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        
        # Grouped or full attention
        if H * W < self.num_groups or H * W <= 1:
            # Full attention for small feature maps
            query_flat = query.view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, H*W, C//8]
            key_flat = key.view(batch_size, -1, H * W)  # [B, C//8, H*W]
            value_flat = value.view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, H*W, C]
            
            attention = torch.bmm(query_flat, key_flat)  # [B, H*W, H*W]
            attention = self.softmax(attention)
            out = torch.bmm(attention, value_flat)  # [B, H*W, C]
            out = out.permute(0, 2, 1).view(batch_size, C, H, W)
            out = self.scale_weight1 * out
        else:
            # Grouped attention for large feature maps
            group_size = H * W // self.num_groups
            query_group = query.view(batch_size, -1, H * W).view(batch_size, -1, self.num_groups, group_size)
            key_group = key.view(batch_size, -1, H * W).view(batch_size, -1, self.num_groups, group_size)
            value_group = value.view(batch_size, -1, H * W).view(batch_size, -1, self.num_groups, group_size)
            
            query_group_reshaped = query_group.permute(0, 2, 3, 1).reshape(batch_size * self.num_groups, group_size, -1)
            key_group_reshaped = key_group.permute(0, 2, 1, 3).reshape(batch_size * self.num_groups, -1, group_size)
            attention = torch.bmm(query_group_reshaped, key_group_reshaped)
            attention = self.softmax(attention)
            
            value_group_reshaped = value_group.permute(0, 2, 1, 3).reshape(batch_size * self.num_groups, -1, group_size)
            out = torch.bmm(value_group_reshaped, attention.transpose(-1, -2))
            out = out.view(batch_size, self.num_groups, -1, group_size).permute(0, 2, 1, 3).reshape(batch_size, C, H, W)
            out = self.scale_weight1 * out
        
        # Multi-scale fusion
        if H < 2 or W < 2:
            # Skip pooling for very small feature maps
            out_pool = torch.zeros_like(x)
        else:
            x_pool = self.pool(x)
            H_pool, W_pool = x_pool.size(2), x_pool.size(3)
            
            # Red-green channel guidance for pooled features
            if self.is_rgb_input and C == 3:
                rg_diff_pool = torch.abs(self.pool(x[:, 0:1, :, :]) - self.pool(x[:, 1:2, :, :]))
            else:
                rgb_proj_pool = self.rg_project(x_pool)
                rg_diff_pool = torch.abs(rgb_proj_pool[:, 0:1, :, :] - rgb_proj_pool[:, 1:2, :, :])
            rg_weight_pool = torch.sigmoid(rg_diff_pool)
            
            # Generate query, key, and value for pooled features
            query_pool = self.query_conv(x_pool * rg_weight_pool)
            key_pool = self.key_conv(x_pool * rg_weight_pool)
            value_pool = self.value_conv(x_pool)
            query_pool = F.normalize(query_pool, dim=1)
            key_pool = F.normalize(key_pool, dim=1)
            
            # Grouped or full attention for pooled features
            if H_pool * W_pool < self.num_groups or H_pool * W_pool <= 1:
                # 整体注意力计算
                query_pool_flat = query_pool.view(batch_size, -1, H_pool * W_pool).permute(0, 2, 1)  # [B, H_pool*W_pool, C//8]
                key_pool_flat = key_pool.view(batch_size, -1, H_pool * W_pool)  # [B, C//8, H_pool*W_pool]
                value_pool_flat = value_pool.view(batch_size, -1, H_pool * W_pool).permute(0, 2, 1)  # [B, H_pool*W_pool, C]
                
                attention_pool = torch.bmm(query_pool_flat, key_pool_flat)  # [B, H_pool*W_pool, H_pool*W_pool]
                attention_pool = self.softmax(attention_pool)
                out_pool = torch.bmm(attention_pool, value_pool_flat)  # [B, H_pool*W_pool, C]
                out_pool = out_pool.permute(0, 2, 1).view(batch_size, C, H_pool, W_pool)
            else:
                group_size_pool = H_pool * W_pool // self.num_groups
                query_pool_group = query_pool.view(batch_size, -1, H_pool * W_pool).view(batch_size, -1, self.num_groups, group_size_pool)
                key_pool_group = key_pool.view(batch_size, -1, H_pool * W_pool).view(batch_size, -1, self.num_groups, group_size_pool)
                value_pool_group = value_pool.view(batch_size, -1, H_pool * W_pool).view(batch_size, -1, self.num_groups, group_size_pool)
                
                query_pool_reshaped = query_pool_group.permute(0, 2, 3, 1).reshape(batch_size * self.num_groups, group_size_pool, -1)
                key_pool_reshaped = key_pool_group.permute(0, 2, 1, 3).reshape(batch_size * self.num_groups, -1, group_size_pool)
                attention_pool = torch.bmm(query_pool_reshaped, key_pool_reshaped)
                attention_pool = self.softmax(attention_pool)
                
                value_pool_reshaped = value_pool_group.permute(0, 2, 1, 3).reshape(batch_size * self.num_groups, -1, group_size_pool)
                out_pool = torch.bmm(value_pool_reshaped, attention_pool.transpose(-1, -2))
                out_pool = out_pool.view(batch_size, self.num_groups, -1, group_size_pool).permute(0, 2, 1, 3).reshape(batch_size, C, H_pool, W_pool)
            
            # Upsample and smooth pooled features
            out_pool = F.interpolate(out_pool, size=(H, W), mode='bilinear', align_corners=False)
            out_pool = self.smooth_conv(out_pool)
            out_pool = self.scale_weight2 * out_pool
        
        # Combine attention and pooled outputs with residual connection
        out = self.gamma * (out + out_pool) + x
        return out


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_size=1024):
        """Initialize the Rotary Position Embedding (RoPE) module.

        This module applies rotary position embeddings to encode positional information in sequences,
        suitable for Transformer-based models in image enhancement tasks, such as red-green color
        vision deficiency (CVD) correction. It precomputes frequency bases and caches cosine and sine
        tables for efficient computation.

        Parameters:
            dim (int): Embedding dimension (must be even, as RoPE splits dimensions into pairs).
            max_size (int, optional): Maximum sequence length for precomputed cache (default: 1024).

        Attributes:
            inv_freq (torch.Tensor): Inverse frequency tensor for rotary embeddings, shape [dim/2].
            max_size (int): Maximum sequence length supported by the cache.
            _cache_cos (torch.Tensor or None): Cached cosine values for rotary transformation.
            _cache_sin (torch.Tensor or None): Cached sine values for rotary transformation.

        Notes:
            - RoPE is inspired by Su et al., "RoFormer: Enhanced Transformer with Rotary Position
              Embedding," 2021, arXiv:2104.09864.
            - In the context of CVD enhancement, this module may be used to encode spatial or sequential
              information in attention mechanisms (e.g., alongside NonLocalBlock).
        """
        super().__init__()
        self.dim = dim
        # Compute inverse frequencies for rotary embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_size = max_size
        self._cache_cos = None
        self._cache_sin = None


    def _update_cos_sin_cache(self, seq_len, device):
        """Update cached cosine and sine tables for rotary position embeddings.

        This internal method computes or updates the cosine and sine tables based on the sequence
        length and device, ensuring efficient computation for rotary transformations.

        Parameters:
            seq_len (int): Length of the input sequence.
            device (torch.device): Device on which to compute the cache (e.g., 'cuda' or 'cpu').

        Notes:
            - The cache is updated only if it is None or too small for the input sequence length.
            - The cosine and sine tables are repeated to match the embedding dimension.
        """
        cache_is_none = self._cache_cos is None
        if cache_is_none:
            cache_too_small = torch.tensor(False)
        else:
            cache_too_small = self._cache_cos.size(0) < torch.tensor(seq_len, device=self._cache_cos.device)
        if cache_is_none or cache_too_small.item():
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = positions.unsqueeze(-1) * self.inv_freq
            self._cache_cos = freqs.cos().repeat(1, 2)
            self._cache_sin = freqs.sin().repeat(1, 2)
    def forward(self, x, seq_len):
        """Apply rotary position embedding to the input tensor.

        This method applies rotary position embeddings to the input tensor by rotating pairs of
        dimensions using precomputed cosine and sine tables. It is designed for Transformer-based
        models in image enhancement tasks, such as red-green color vision deficiency correction.

        Parameters:
            x (torch.Tensor): Input tensor, shape [batch, seq_len, dim], typically in [-1, 1].
            seq_len (int): Length of the input sequence.

        Returns:
            torch.Tensor: Output tensor with rotary position embeddings applied, same shape as input.

        Notes:
            - The input tensor is split into pairs of dimensions (x1, x2), which are rotated using
              cosine and sine transformations.
            - The method assumes an even embedding dimension (dim) for pairwise rotation.
        """
        self._update_cos_sin_cache(seq_len, x.device)
        cos = self._cache_cos[:seq_len].unsqueeze(0)  # (1, seq_len, dim)
        sin = self._cache_sin[:seq_len].unsqueeze(0)  # (1, seq_len, dim)
        # Split input into even and odd dimensions for rotation
        x1, x2 = x[..., 0::2], x[..., 1::2]  # [batch, seq_len, dim/2]
        x_rot = torch.stack([-x2, x1], dim=-1).reshape(x.shape)  # Apply rotation
        return x * cos + x_rot * sin
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4, mlp_ratio=4.0, drop=0.5):
        """Initialize the Window-based Multi-head Self-Attention (W-MSA) module.

        This module implements window-based multi-head self-attention with residual connections,
        LayerNorm, and Rotary Position Embedding (RoPE) for image enhancement tasks, such as
        red-green color vision deficiency (CVD) correction. It processes input feature maps within
        fixed-size windows to capture local dependencies efficiently.

        Parameters:
            dim (int): Number of input feature channels.
            window_size (int, optional): Size of the attention window (default: 4).
            num_heads (int, optional): Number of attention heads (default: 4).
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension (default: 4.0).
            drop (float, optional): Dropout rate for attention and MLP layers (default: 0.5).

        Attributes:
            norm1 (nn.LayerNorm): Layer normalization before attention.
            norm2 (nn.LayerNorm): Layer normalization before MLP.
            qkv (nn.Linear): Linear layer to generate query, key, and value.
            proj (nn.Linear): Linear layer for output projection.
            rope (RotaryPositionEmbedding): Rotary position embedding module.
            attn_drop (nn.Dropout): Dropout for attention weights.
            mlp (nn.Sequential): MLP with GELU activation and dropout.
            softmax (nn.Softmax): Softmax for attention weights.
            smooth_conv (nn.Conv2d): 3x3 convolution for smoothing output.
            scale (float): Scaling factor for attention scores.

        Notes:
            - Inspired by Swin Transformer (Liu et al., "Swin Transformer: Hierarchical Vision Transformer
              using Shifted Windows," ICCV 2021).
            - Used in CVD enhancement to capture local color and structural dependencies within windows.
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear layers for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Rotary position embedding
        self.rope = RotaryPositionEmbedding(dim // num_heads)

        # Attention dropout
        self.attn_drop = nn.Dropout(drop)

        # MLP module
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

        # Softmax and smoothing convolution
        self.softmax = nn.Softmax(dim=-1)
        self.smooth_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        """Apply window-based multi-head self-attention (W-MSA) to the input tensor.

        This method processes the input feature map by dividing it into non-overlapping windows,
        applying multi-head self-attention with rotary position embeddings within each window,
        and combining with residual connections and MLP. A smoothing convolution is applied to the output.

        Parameters:
            x (torch.Tensor): Input tensor, shape [B, C, H, W], values typically in [-1, 1].

        Returns:
            torch.Tensor: Output tensor, shape [B, C, H, W], with attention-enhanced features.

        Notes:
            - Padding is applied to handle non-divisible input sizes.
            - The module uses rotary position embeddings (RoPE) to encode spatial positions within windows.
            - A 3x3 convolution is applied post-attention to smooth the output.
        """
        B, C, H, W = x.shape
        M = self.window_size

        # Pad input to support non-divisible sizes
        pad_h = (M - H % M) % M
        pad_w = (M - W % M) % M
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H_padded, W_padded = x.shape

        # Divide into windows
        num_h, num_w = H_padded // M, W_padded // M
        x = x.view(B, C, num_h, M, num_w, M).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * num_h * num_w, C, M * M).transpose(1, 2)  # (B * num_windows, M*M, C)

        # Residual connection: normalize input
        shortcut = x
        x = self.norm1(x)

        # Generate query, key, value
        qkv = self.qkv(x).reshape(B * num_h * num_w, M * M, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B * num_windows, num_heads, M*M, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary position embedding
        q = self.rope(q, M * M)  # (B * num_windows, num_heads, M*M, dim_head)
        k = self.rope(k, M * M)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B * num_windows, num_heads, M*M, M*M)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B * num_h * num_w, M * M, C)

        # Output projection
        x = self.proj(x)

        # Residual connection 1: attention output + input
        x = shortcut + x

        # Residual connection 2: MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x

        # Reshape and remove padding
        x = x.view(B, num_h, num_w, M, M, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H_padded, W_padded)[:, :, :H, :W]
        x = self.smooth_conv(x)  # Apply smoothing convolution
        return x
class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4, shift_size=2, mlp_ratio=4.0, drop=0.5):
        """Initialize the Shifted Window-based Multi-head Self-Attention (SW-MSA) module.

        This module extends W-MSA by applying a cyclic shift to the input feature map, enabling
        cross-window interactions for enhanced feature representation in image enhancement tasks,
        such as red-green color vision deficiency (CVD) correction.

        Parameters:
            dim (int): Number of input feature channels.
            window_size (int, optional): Size of the attention window (default: 4).
            num_heads (int, optional): Number of attention heads (default: 4).
            shift_size (int, optional): Cyclic shift size, typically half the window size (default: 2).
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension (default: 4.0).
            drop (float, optional): Dropout rate for attention and MLP layers (default: 0.5).

        Attributes:
            wmsa (WindowAttention): W-MSA module for attention computation.
            window_size (int): Size of the attention window.
            shift_size (int): Cyclic shift size.

        Notes:
            - Inspired by Swin Transformer (Liu et al., "Swin Transformer: Hierarchical Vision Transformer
              using Shifted Windows," ICCV 2021).
            - The cyclic shift enhances connectivity between windows, improving global context for CVD enhancement.
        """
        super(ShiftedWindowAttention, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.wmsa = WindowAttention(dim, window_size, num_heads, mlp_ratio, drop)

    def forward(self, x):
        """Apply shifted window-based multi-head self-attention (SW-MSA) to the input tensor.

        This method applies a cyclic shift to the input feature map, processes it with W-MSA, and
        reverses the shift to produce attention-enhanced features. It is designed for image enhancement
        tasks, such as red-green color vision deficiency correction.

        Parameters:
            x (torch.Tensor): Input tensor, shape [B, C, H, W], values typically in [-1, 1].

        Returns:
            torch.Tensor: Output tensor, shape [B, C, H, W], with attention-enhanced features.

        Notes:
            - Padding is applied to handle non-divisible input sizes.
            - The cyclic shift enables cross-window interactions, capturing broader spatial context.
        """
        B, C, H, W = x.shape

        # 动态计算 padding 以支持移位
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H_padded, W_padded = x.shape

        # 移位操作
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # 使用 W-MSA 计算注意力
        attn_output = self.wmsa(shifted_x)

        # 逆移位
        attn_output = torch.roll(attn_output, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # 裁剪 padding
        attn_output = attn_output[:, :, :H, :W]
        return attn_output


class PerceptualLoss(nn.Module):
    def __init__(self, layers=None, layer_weights=None):
        """Initialize the Perceptual Loss module using VGG19 features.

        This module computes the perceptual loss between predicted and target images by comparing
        features extracted from specific VGG19 layers. It is used in red-green color vision deficiency
        (CVD) enhancement to ensure perceptual similarity between enhanced and target images.

        Parameters:
            layers (list of int, optional): Indices of VGG19 layers to extract features (default: [0, 5, 10]).
            layer_weights (list of float, optional): Weights for each layer's loss (default: [1.0, 1.0, 1.0]).

        Attributes:
            vgg (nn.Module): Pretrained VGG19 feature extractor (frozen).
            layers (list): Indices of VGG19 layers to use.
            layer_weights (list): Weights for each layer's loss.
            mean (torch.Tensor): Mean for VGG19 input normalization, shape [1, 3, 1, 1].
            std (torch.Tensor): Standard deviation for VGG19 input normalization, shape [1, 3, 1, 1].

        Notes:
            - Uses VGG19 pretrained on ImageNet, with features from low to mid-level layers (conv1_1, conv2_2, conv3_3).
            - Input images are normalized to match VGG19's expected input range.
        """
        super(PerceptualLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg = models.vgg19(pretrained=True).features.eval().to(device)
        self.vgg = vgg

        if layers is None:
            self.layers = [0, 5, 10]
        else:
            self.layers = layers
        
        if layer_weights is None:
            self.layer_weights = [1.0, 1.0, 1.0]
        else:
            self.layer_weights = layer_weights
        assert len(self.layers) == len(self.layer_weights), 

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        
    def forward(self, prediction, target):
        """Compute perceptual loss between predicted and target images.

        This method extracts features from specified VGG19 layers for both predicted and target images,
        computing the weighted mean squared error (MSE) between them. It is used to ensure perceptual
        similarity in red-green color vision deficiency enhancement.

        Parameters:
            prediction (torch.Tensor): Predicted image tensor, shape [B, 3, H, W], values in [-1, 1].
            target (torch.Tensor): Target image tensor, shape [B, 3, H, W], values in [-1, 1].

        Returns:
            torch.Tensor: Scalar perceptual loss value.

        Notes:
            - Inputs are normalized to [0, 1] and then to VGG19's expected range before feature extraction.
            - The loss is a weighted sum of MSE losses from specified VGG19 layers.
        """
        prediction = (prediction + 1) / 2.0
        target = (target + 1) / 2.0

        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        pred_features = self.get_features(prediction)
        target_features = self.get_features(target)

        loss = 0
        for p_feat, t_feat, weight in zip(pred_features, target_features , self.layer_weights):
            loss += weight * F.mse_loss(p_feat, t_feat)
        return loss

    def get_features(self, x):
        """Extract features from specified VGG19 layers.

        Parameters:
            x (torch.Tensor): Input tensor, shape [B, 3, H, W], normalized for VGG19.

        Returns:
            list of torch.Tensor: Features extracted from specified VGG19 layers.
        """
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real,real_image=None, fake_image=None):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False, use_nonlocalblock=True, layer_idx=None):
        """Initialize a U-Net skip connection block for red-green color vision deficiency enhancement.

        Constructs a U-Net block with downsampling, upsampling, and optional attention mechanisms
        (NonLocalBlock, WindowAttention, or ShiftedWindowAttention) to enhance color and structural
        features for red-green color vision deficiency (CVD) correction.

        Parameters:
            outer_nc (int): Number of output channels.
            inner_nc (int): Number of input channels for the inner layer.
            input_nc (int, optional): Number of input channels (defaults to outer_nc if None).
            submodule (nn.Module, optional): Nested U-Net block (default: None).
            outermost (bool, optional): If True, this is the outermost block (default: False).
            innermost (bool, optional): If True, this is the innermost block (default: False).
            norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm2d).
            use_dropout (bool, optional): If True, apply dropout in non-outermost blocks (default: False).
            use_attention (bool, optional): If True, use attention mechanisms (default: False).
            use_nonlocalblock (bool, optional): If True, use NonLocalBlock for deep layers (default: True).
            layer_idx (int, optional): Layer index for selecting attention type (default: None).

        Attributes:
            model (nn.Sequential): Sequential model of downsampling, submodule, and upsampling layers.
            attention (nn.Module, optional): Attention module for downsampling path.
            decoder_attention (nn.Module, optional): Attention module for upsampling path.
            outermost (bool): Whether this is the outermost block.
            use_attention (bool): Whether attention is enabled.
            use_nonlocalblock (bool): Whether NonLocalBlock is enabled.
            layer_idx (int): Layer index for attention selection.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.use_attention = use_attention
        self.use_nonlocalblock = use_nonlocalblock  # 是否使用NonLocalBlock
        self.layer_idx = layer_idx

        # Determine bias usage based on normalization layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        # Downsampling layers
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        # Upsampling layers
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        if self.use_attention:
            # Select attention mechanism based on layer index and non-local block flag
            if self.layer_idx < 3 and self.use_nonlocalblock:  # 深层使用NonLocalBlock
                self.attention = NonLocalBlock(inner_nc, num_groups=4, is_rgb_input=(input_nc == 3))
                self.decoder_attention = NonLocalBlock(outer_nc, num_groups=4, is_rgb_input=False)
            else:  # 浅层交替使用W-MSA和SW-MSA
                self.attention = ShiftedWindowAttention(inner_nc, window_size=4, num_heads=4, shift_size=2) \
                    if layer_idx % 2 == 0 else WindowAttention(inner_nc, window_size=4, num_heads=4)
                self.decoder_attention = ShiftedWindowAttention(outer_nc, window_size=4, num_heads=4, shift_size=2) \
                    if layer_idx % 2 == 0 else WindowAttention(outer_nc, window_size=4, num_heads=4)
        

        # Construct model based on block type
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if self.use_attention:
                down += [self.attention]
            up = [uprelu, upconv]
            if self.use_attention:
                up += [self.decoder_attention]
            up += [upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if self.use_attention:
                down += [self.attention]
            down += [downnorm]
            up = [uprelu, upconv]
            if self.use_attention:
                up += [self.decoder_attention]
            up += [upnorm]
            
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)


    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True):
        """Initialize the U-Net generator for red-green color vision deficiency enhancement.

        Constructs a U-Net with skip connections, optionally incorporating attention mechanisms
        (e.g., WindowAttention) and non-local blocks (e.g., NonLocalBlock) to enhance color and
        structural features for red-green color vision deficiency (CVD) correction.

        Parameters:
            input_nc (int): Number of input channels (e.g., 3 for RGB).
            output_nc (int): Number of output channels (e.g., 3 for enhanced RGB).
            num_downs (int): Number of downsampling layers in the U-Net.
            ngf (int, optional): Number of filters in the first layer (default: 64).
            norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm2d).
            use_dropout (bool, optional): If True, apply dropout in inner layers (default: False).
            use_attention (bool, optional): If True, use attention mechanisms (default: True).
            use_nonlocalblock (bool, optional): If True, use non-local blocks (default: True).

        Attributes:
            model (UnetSkipConnectionBlock): The complete U-Net model with nested submodules.

        Notes:
            - The U-Net architecture is based on Ronneberger et al., "U-Net: Convolutional Networks
              for Biomedical Image Segmentation," MICCAI 2015.
            - Attention and non-local blocks enhance feature representation for CVD correction.
        """
        super(UnetGenerator, self).__init__()
        # Innermost block
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                            innermost=True, use_attention=True, use_nonlocalblock=True, layer_idx=0)
        layer_idx = 1
        # Inner blocks with attention and non-local blocks
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer, use_dropout=use_dropout,
                                                use_attention=True, use_nonlocalblock=True, layer_idx=layer_idx)
            layer_idx += 1
        # Middle blocks
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                            norm_layer=norm_layer, use_attention=True, use_nonlocalblock=False, layer_idx=layer_idx)
        layer_idx += 1
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                            norm_layer=norm_layer, use_attention=True, use_nonlocalblock=False, layer_idx=layer_idx)
        layer_idx += 1
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                            norm_layer=norm_layer, use_attention=True, use_nonlocalblock=False, layer_idx=layer_idx)
        layer_idx += 1
        # Outermost block
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                            outermost=True, norm_layer=norm_layer, use_attention=False, layer_idx=layer_idx)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
