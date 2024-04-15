import math

import torch
from torch import nn

from models.modules import ConvNextBlock, SinusoidalPosEmb


class NextNet(nn.Module):
    """
    A backbone model comprised of a chain of ConvNext blocks, with skip connections.
    The skip connections are connected similar to a "U-Net" structure (first to last, middle to middle, etc).
    """
    def __init__(self, in_channels=3, out_channels=3, depth=16, filters_per_layer=64, frame_conditioned=False):
        """
        Args:
            in_channels (int):
                Number of input image channels.
            out_channels (int):
                Number of network output channels.
            depth (int):
                Number of ConvNext blocks in the network.
            filters_per_layer (int):
                Base dimension in each ConvNext block.
            frame_conditioned (bool):
                Whether to condition the network on the difference between the current and previous frames. Should
                be True when training a DDPM frame predictor.
        """
        super().__init__()

        if isinstance(filters_per_layer, (list, tuple)):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        time_dim = dims[0]
        emb_dim = time_dim * 3 if frame_conditioned else time_dim
# ---
        # robot_emb_dim = emb_dim
# ---
        self.depth = depth
        self.layers = nn.ModuleList([])
# ---
        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(in_channels, dims[0], emb_dim=emb_dim, norm=False))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims[i - 1], dims[i], emb_dim=emb_dim,  norm=True))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNextBlock(2 * dims[i - 1], dims[i], emb_dim=emb_dim, norm=True)) # x and x from 'encoding' procedure concatenated, hence why 2*dims[i-1]
# ---
        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.Conv2d(dims[depth - 1], out_channels, 1)

        # Encoder for positional embedding of timestep
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
# ---
        # Encoder for linear embedding of robot pose
        # self.robot_encoder = nn.Linear(robot_emb_dim, emb_dim)
        self.robot_encoder = nn.Sequential(
            nn.Linear(6, time_dim),
            nn.ReLU(), 
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )        
# ---
        if frame_conditioned:
            # Encoder for positional embedding of frame
            self.frame_encoder = nn.Sequential(
                 SinusoidalPosEmb(time_dim),
                 nn.Linear(time_dim, time_dim * 4),
                 nn.GELU(),
                 nn.Linear(time_dim * 4, time_dim)
            )

    def forward(self, x, t, robot_pose=None, frame_diff=None):
        time_embedding = self.time_encoder(t)
        # print(f'time_embedding is {time_embedding.shape}?????????????????')
# ---
        # print(f'+++++++++++{time_embedding}+++++++++++')
        # print(f'==========={t} and {robot_pose} and {frame_diff} =============')
        if robot_pose is not None:
            robot_embedding = self.robot_encoder(robot_pose)
            # print(f'{robot_embedding.shape}?????????????????')
            embedding = torch.cat([time_embedding, robot_embedding], dim=1)
        else:
            embedding = time_embedding
# ---
        if frame_diff is not None:
            # print('what the funccccccccccccccccccccccccccccccccccc')
            frame_embedding = self.frame_encoder(frame_diff)
            embedding = torch.cat([embedding, frame_embedding], dim=1)
        else:
            embedding = embedding

        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, embedding)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, embedding)

        return self.final_conv(x)
