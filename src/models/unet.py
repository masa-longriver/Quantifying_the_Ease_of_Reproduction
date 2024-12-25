import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv(
    config: dict,
    in_channel: int,
    out_channel: int,
    stride: int = None,
    bias: bool = True,
    dilation: int = None,
    init_scale: float = None,
    padding: int = None
) -> nn.Conv2d:
    """
    Creates a 3x3 convolutional layer with specified parameters.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to None.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Defaults to True.
        dilation (int, optional): Spacing between kernel elements. Defaults to
            None.
        init_scale (float, optional): Scale for weight initialization. Defaults
            to None.
        padding (int, optional): Zero-padding added to both sides of the input.
            Defaults to None.

    Returns:
        nn.Conv2d: Configured 2D convolutional layer.
    """
    conv_config = config['unet']['conv']
    if stride is None:
        stride = conv_config['stride']
    if dilation is None:
        dilation = conv_config['dilation']
    if init_scale is None:
        init_scale = conv_config['init_scale']
    elif init_scale == 0:
        init_scale = 1e-10
    if padding is None:
        padding = conv_config['padding']

    conv = nn.Conv2d(
        in_channel,
        out_channel,
        stride=stride,
        bias=bias,
        dilation=dilation,
        padding=padding,
        kernel_size=conv_config['kernel_size']
    ).to(config['device'])
    nn.init.xavier_uniform_(conv.weight, gain=init_scale**2)
    nn.init.zeros_(conv.bias)

    return conv


class TimeEmbedding(nn.Module):
    """
    Time embedding using the same method as Transformer, then passing through
    a Multi-Layer Perceptron (MLP).

    Args:
        config (dict): Configuration dictionary containing model parameters.
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.emb_config = config['unet']['time_embedding']
        self.linear_1 = nn.Linear(
            self.emb_config['emb_dim1'],
            self.emb_config['emb_dim2']
        ).to(config['device'])
        self.linear_2 = nn.Linear(
            self.emb_config['emb_dim2'],
            self.emb_config['emb_dim2']
        ).to(config['device'])
        self.act = nn.SiLU()

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the time embedding.

        Args:
            t (torch.Tensor): Input tensor representing time steps.

        Returns:
            torch.Tensor: Output tensor after applying time embedding and
            passing through MLP.
        """
        half_dim = self.emb_config['emb_dim1'] // 2
        emb = math.log(self.emb_config['max_positions']) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device)
            * -emb
        )
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        emb = self.linear_1(emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)

        return emb


class NIN(nn.Module):
    """
    Network-in-Network (NIN) layer.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        input_dim (int): Dimension of the input features.
        num_units (int): Number of units in the NIN layer.
        init_scale (float, optional): Scale for weight initialization.
            Defaults to 1.
    """
    def __init__(
        self,
        config: dict,
        input_dim: int,
        num_units: int,
        init_scale: float = 1.
    ) -> None:
        super().__init__()
        if init_scale == 0:
            init_scale = 1e-10
        self.W = nn.Parameter(
            torch.empty(input_dim, num_units),
            requires_grad=True
        ).to(config['device'])
        self.b = nn.Parameter(
            torch.zeros(num_units), requires_grad=True
        ).to(config['device'])
        nn.init.xavier_uniform_(self.W, gain=init_scale**2)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Network-in-Network (NIN) layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels,
            height, width).

        Returns:
            torch.Tensor: Output tensor after applying the NIN layer, with
            shape (batch_size, num_units, height, width).
        """
        x = x.permute(0, 2, 3, 1)
        y = torch.einsum("...i,ij->...j", x, self.W) + self.b
        y = y.permute(0, 3, 1, 2)

        return y


class ResnetBlock(nn.Module):
    """
    Residual block with Group Normalization and SiLU activation.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """
    def __init__(
        self,
        config: dict,
        in_channel: int,
        out_channel: int
    ) -> None:
        super().__init__()
        model_config = config['unet']
        self.out_channel = out_channel
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=model_config['groupnorm']['num_groups'],
            num_channels=in_channel,
            eps=1e-6
        ).to(config['device'])
        self.act = nn.SiLU()
        self.Conv_1 = Conv(config, in_channel, out_channel, init_scale=0.)
        self.Dense = nn.Linear(
            model_config['time_embedding']['emb_dim2'], out_channel
        ).to(config['device'])
        self.GroupNorm_2 = nn.GroupNorm(
            num_groups=model_config['groupnorm']['num_groups'],
            num_channels=out_channel,
            eps=1e-6
        ).to(config['device'])
        self.Dropout = nn.Dropout(model_config['dropout'])
        self.Conv_2 = Conv(config, out_channel, out_channel)
        self.NIN = NIN(config, in_channel, out_channel)

        nn.init.xavier_uniform_(self.Dense.weight)
        nn.init.zeros_(self.Dense.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResnetBlock.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels,
            height, width).
            t_emb (torch.Tensor): Time embedding tensor with shape (batch_size,
            emb_dim2).

        Returns:
            torch.Tensor: Output tensor after applying the ResnetBlock.
        """
        B, C, H, W = x.shape
        h = self.GroupNorm_1(x)
        h = self.act(h)
        h = self.Conv_1(h) + self.Dense(t_emb)[:, :, None, None]
        h = self.GroupNorm_2(h)
        h = self.act(h)
        h = self.Dropout(h)
        h = self.Conv_2(h)
        if C != self.out_channel:
            x = self.NIN(x)

        return h + x


class AttnBlock(nn.Module):
    """
    Attention block with Group Normalization and SiLU activation.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        channels (int): Number of channels in the input tensor.
    """
    def __init__(self, config: dict, channels: int) -> None:
        super().__init__()
        model_config = config['unet']
        self.GroupNorm = nn.GroupNorm(
            num_groups=model_config['groupnorm']['num_groups'],
            num_channels=channels,
            eps=1e-6
        ).to(config['device'])
        self.NIN_q = NIN(config, channels, channels)
        self.NIN_k = NIN(config, channels, channels)
        self.NIN_v = NIN(config, channels, channels)
        self.NIN_tot = NIN(config, channels, channels, init_scale=0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Attention Block.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels,
            height, width).

        Returns:
            torch.Tensor: Output tensor after applying the Attention Block.
        """
        B, C, H, W = x.shape
        h = self.GroupNorm(x)
        q = self.NIN_q(h)
        k = self.NIN_k(h)
        v = self.NIN_v(h)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_tot(h)

        return h + x


class Upsample(nn.Module):
    """
    Upsample layer with 3x3 convolution.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        channels (int): Number of channels in the input tensor.
    """
    def __init__(self, config: dict, channels: int) -> None:
        super().__init__()
        self.Conv = Conv(config, channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Upsample layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels,
            height, width).

        Returns:
            torch.Tensor: Output tensor after applying the Upsample layer.
        """
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), mode='nearest')
        h = self.Conv(h)

        return h


class DownSample(nn.Module):
    """
    Downsample layer with 3x3 convolution.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        channels (int): Number of channels in the input tensor.
    """
    def __init__(self, config: dict, channels: int) -> None:
        super().__init__()
        self.Conv = Conv(config, channels, channels, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Downsample layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels,
            height, width).

        Returns:
            torch.Tensor: Output tensor after applying the Downsample layer.
        """
        h = F.pad(x, (0, 1, 0, 1))
        h = self.Conv(h)

        return h


class UNet(nn.Module):
    """
    U-Net architecture for score-based diffusion models.

    Args:
        config (dict): Configuration dictionary containing model parameters.
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config['unet']
        self.Conv1 = Conv(
            config,
            config['data']['channel'],
            model_config['channel1']
        )
        self.TimeEmbedding = TimeEmbedding(config)

        # DownBlock1
        self.ResnetBlock_D11 = ResnetBlock(
            config, model_config['channel1'], model_config['channel1']
        )
        self.ResnetBlock_D12 = ResnetBlock(
            config, model_config['channel1'], model_config['channel1']
        )
        self.Downsample_1 = DownSample(config, model_config['channel1'])

        # DownBlock2
        self.ResnetBlock_D21 = ResnetBlock(
            config, model_config['channel1'], model_config['channel2']
        )
        self.AttnBlock_D21 = AttnBlock(config, model_config['channel2'])
        self.ResnetBlock_D22 = ResnetBlock(
            config, model_config['channel2'], model_config['channel2']
        )
        self.AttnBlock_D22 = AttnBlock(config, model_config['channel2'])
        self.Downsample_2 = DownSample(config, model_config['channel2'])

        # DownBlock3
        self.ResnetBlock_D31 = ResnetBlock(
            config, model_config['channel2'], model_config['channel2']
        )
        self.ResnetBlock_D32 = ResnetBlock(
            config, model_config['channel2'], model_config['channel2']
        )
        self.Downsample_3 = DownSample(config, model_config['channel2'])

        # DownBlock4
        self.ResnetBlock_D41 = ResnetBlock(
            config, model_config['channel2'], model_config['channel2']
        )
        self.ResnetBlock_D42 = ResnetBlock(
            config, model_config['channel2'], model_config['channel2']
        )

        # CenterBlock
        self.ResnetBlock_C1 = ResnetBlock(
            config, model_config['channel2'], model_config['channel2']
        )
        self.AttnBlock_C = AttnBlock(config, model_config['channel2'])
        self.ResnetBlock_C2 = ResnetBlock(
            config, model_config['channel2'], model_config['channel2']
        )

        # UpBlock1
        self.ResnetBlock_U11 = ResnetBlock(
            config, model_config['channel2'] * 2, model_config['channel2']
        )
        self.ResnetBlock_U12 = ResnetBlock(
            config, model_config['channel2'] * 2, model_config['channel2']
        )
        self.ResnetBlock_U13 = ResnetBlock(
            config, model_config['channel2'] * 2, model_config['channel2']
        )
        self.Upsample_1 = Upsample(config, model_config['channel2'])

        # UpBlock2
        self.ResnetBlock_U21 = ResnetBlock(
            config, model_config['channel2'] * 2, model_config['channel2']
        )
        self.ResnetBlock_U22 = ResnetBlock(
            config, model_config['channel2'] * 2, model_config['channel2']
        )
        self.ResnetBlock_U23 = ResnetBlock(
            config, model_config['channel2'] * 2, model_config['channel2']
        )
        self.Upsample_2 = Upsample(config, model_config['channel2'])

        # UpBlock3
        self.ResnetBlock_U31 = ResnetBlock(
            config, model_config['channel2'] * 2, model_config['channel2']
        )
        self.ResnetBlock_U32 = ResnetBlock(
            config, model_config['channel2'] * 2, model_config['channel2']
        )
        self.ResnetBlock_U33 = ResnetBlock(
            config,
            model_config['channel2'] + model_config['channel1'],
            model_config['channel2']
        )
        self.AttnBlock_U3 = AttnBlock(config, model_config['channel2'])
        self.Upsample_3 = Upsample(config, model_config['channel2'])

        # UpBlock4
        self.ResnetBlock_U41 = ResnetBlock(
            config,
            model_config['channel2'] + model_config['channel1'],
            model_config['channel1']
        )
        self.ResnetBlock_U42 = ResnetBlock(
            config, model_config['channel1'] * 2, model_config['channel1']
        )
        self.ResnetBlock_U43 = ResnetBlock(
            config, model_config['channel1'] * 2, model_config['channel1']
        )

        self.GroupNorm = nn.GroupNorm(
            num_groups=model_config['groupnorm']['num_groups'],
            num_channels=model_config['channel1'],
            eps=1e-6
        ).to(config['device'])
        self.act = nn.SiLU()
        self.Conv2 = Conv(
            config, model_config['channel1'], config['data']['channel']
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the U-Net.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels,
                height, width).
            t_emb (torch.Tensor): Time embedding tensor with shape (batch_size,
                emb_dim2).

        Returns:
            torch.Tensor: Output tensor after applying the U-Net.
        """
        h_0 = self.Conv1(x)
        t_emb = self.TimeEmbedding(t_emb)
        # DownBlock1
        h_1 = self.ResnetBlock_D11(h_0, t_emb)
        h_2 = self.ResnetBlock_D12(h_1, t_emb)
        h_3 = self.Downsample_1(h_2)
        # DownBlock2
        h_4 = self.ResnetBlock_D21(h_3, t_emb)
        h_4 = self.AttnBlock_D21(h_4)
        h_5 = self.ResnetBlock_D22(h_4, t_emb)
        h_5 = self.AttnBlock_D22(h_5)
        h_6 = self.Downsample_2(h_5)
        # DownBlock3
        h_7 = self.ResnetBlock_D31(h_6, t_emb)
        h_8 = self.ResnetBlock_D32(h_7, t_emb)
        h_9 = self.Downsample_3(h_8)
        # DownBlock4
        h_10 = self.ResnetBlock_D41(h_9, t_emb)
        h_11 = self.ResnetBlock_D42(h_10, t_emb)
        # CenterBlock
        h_12 = self.ResnetBlock_C1(h_11, t_emb)
        h_12 = self.AttnBlock_C(h_12)
        h_12 = self.ResnetBlock_C2(h_12, t_emb)
        # UpBlock1
        h_11 = self.ResnetBlock_U11(torch.cat([h_12, h_11], dim=1), t_emb)
        h_10 = self.ResnetBlock_U12(torch.cat([h_11, h_10], dim=1), t_emb)
        h_9 = self.ResnetBlock_U13(torch.cat([h_10, h_9], dim=1), t_emb)
        h_9 = self.Upsample_1(h_9)
        # UpBlock2
        h_8 = self.ResnetBlock_U21(torch.cat([h_9, h_8], dim=1), t_emb)
        h_7 = self.ResnetBlock_U22(torch.cat([h_8, h_7], dim=1), t_emb)
        h_6 = self.ResnetBlock_U23(torch.cat([h_7, h_6], dim=1), t_emb)
        h_6 = self.Upsample_2(h_6)
        # UpBlock3
        h_5 = self.ResnetBlock_U31(torch.cat([h_6, h_5], dim=1), t_emb)
        h_4 = self.ResnetBlock_U32(torch.cat([h_5, h_4], dim=1), t_emb)
        h_3 = self.ResnetBlock_U33(torch.cat([h_4, h_3], dim=1), t_emb)
        h_3 = self.AttnBlock_U3(h_3)
        h_3 = self.Upsample_3(h_3)
        # UpBlock4
        h_2 = self.ResnetBlock_U41(torch.cat([h_3, h_2], dim=1), t_emb)
        h_1 = self.ResnetBlock_U42(torch.cat([h_2, h_1], dim=1), t_emb)
        h_0 = self.ResnetBlock_U43(torch.cat([h_1, h_0], dim=1), t_emb)

        h_0 = self.GroupNorm(h_0)
        h_0 = self.act(h_0)
        h_0 = self.Conv2(h_0)

        return h_0
