"""
Decoder network for Hierarchical VAE.

Reconstructs input from hierarchical latent representations.
"""

import torch
import torch.nn as nn


class HierarchicalDecoder(nn.Module):
    """
    Hierarchical decoder that reconstructs from concatenated latents.
    
    Args:
        output_dim (int): Output dimension (e.g., 4096 for 1024bp one-hot)
        latent_dims (list): Dimensions for each latent level [L1, L2, L3]
        dropout (float): Dropout probability
        
    Example:
        >>> decoder = HierarchicalDecoder(4096, [256, 512, 1024])
        >>> latents = (torch.randn(32, 256), torch.randn(32, 512), torch.randn(32, 1024))
        >>> reconstruction = decoder(latents)
    """
    
    def __init__(self, output_dim=4096, latent_dims=None, dropout=0.3):
        super().__init__()
        
        if latent_dims is None:
            latent_dims = [256, 512, 1024]
        
        total_latent_dim = sum(latent_dims)
        
        # Decoder stages
        self.dec1 = nn.Sequential(
            nn.Linear(total_latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dec2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dec3 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Linear(2048, output_dim)
    
    def forward(self, latents):
        """
        Decode from hierarchical latents to reconstruction.
        
        Args:
            latents (tuple): (z1, z2, z3) latent vectors
            
        Returns:
            Tensor: Reconstructed input
        """
        # Concatenate all latent levels
        z = torch.cat(latents, dim=-1)
        
        # Decode
        h = self.dec1(z)
        h = self.dec2(h)
        h = self.dec3(h)
        
        return self.output(h)