import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_seq_length: int, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model (int): Embedding dimension of the model
            max_seq_length (int): Maximum sequence length to consider
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)




class EVChargingDecoder(nn.Module):
    """Decoder for EV charging data imputation with uncertainty estimation."""

    def __init__(
        self, 
        d_emb: int,             # LLM embedding dimension
        d_latent: int,          # Latent space dimension
        d_model: int,           # Transformer model dimension
        num_layers: int,        # Number of transformer layers
        nhead: int,             # Number of attention heads
        d_station: int,         # Station embedding dimension
        d_calendar: int,        # Calendar embedding dimension
        num_stations: int,      # number of unique stations
        calendar_feature_dim: int, # Dimension of calendar features
        seq_length: int,        # Sequence length (7 days)
        output_dim: int,        # Output dimension (typically 1 for demand)
        dropout: float=0.1,    # Dropout probability
    ):
        """
        Initialize the EV charging decoder.

        Args:
            d_emb (int): LLM embedding dimension
            d_latent (int): Latent space dimension
            d_model (int): Transformer model dimension
            num_layers (int): Number of transformer layers
            nhead (int): Number of attention heads
            d_station (int): Station embedding dimension
            d_calendar (int): Calendar embedding dimension
            station_feature_dim (int): 
            calendar_feature_dim (int): Dimension of calendar features
            seq_length (int): Sequence length (7 days)
            output_dim (int): Output dimension (typically 1 for demand)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        
        # 1. Variational Component
        self.mu_net = nn.Linear(d_emb, d_latent)
        self.logvar_net = nn.Linear(d_emb, d_latent)
        
        # 2. Station and Calendar Embeddings
        self.station_embedding = nn.Embedding(num_stations, d_station)
        self.calendar_embedding = nn.Linear(calendar_feature_dim, d_calendar)
        
        # 3. Conditioning Network
        self.conditioning_net = nn.Sequential(
            nn.Linear(d_latent + d_station + d_calendar, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2)
        )
        
        # 4. Pure Decoder-Only Transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        self.positional_encoding = PositionalEncoding(d_model, seq_length, dropout)
        
        # Initial token projection
        self.token_embedding = nn.Linear(1, d_model)
        
        # 5. Output projections with uncertainty
        self.mean_output = nn.Linear(d_model, output_dim)
        self.logvar_output = nn.Linear(d_model, output_dim)
        
        # Store dimensions
        self.d_model = d_model
        self.seq_length = seq_length
        self.output_dim = output_dim
        


    def encode(self, llm_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode LLM embedding to variational parameters.

        Args:
            llm_embedding (torch.Tensor): LLM embedding tensor [batch_size, d_emb]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance tensors
        """
        
        mu = self.mu_net(llm_embedding)
        logvar = self.logvar_net(llm_embedding).clamp(-5, 2)
        
        return mu, logvar
    
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize for sampling from latent distribution.

        Args:
            mu (torch.Tensor): Mean tensor [batch_size, d_latent]
            logvar (torch.Tensor): Log variance tensor [batch_size, d_latent]

        Returns:
            torch.Tensor: Sampled latent vector [batch_size, d_latent]
        """
        if self.training:
            logvar = torch.clamp(logvar, -5, 2)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
        
    
    def condition_with_metadata(
        self, 
        z: torch.Tensor, 
        station_id: torch.Tensor, 
        calendar_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Condition the latent representation with station and calendar metadata.

        Args:
            z (torch.Tensor): Latent vector [batch_size, d_latent]
            station_id (torch.Tensor): Station IDs [batch_size]
            calendar_features (torch.Tensor): Calendar feature tensor [batch_size, calendar_feature_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gamma and beta conditioning parameters
        """
        # Get embeddings
        station_emb = self.station_embedding(station_id)
        calendar_emb = self.calendar_embedding(calendar_features)
        
        # Combine all information
        combined = torch.cat([z, station_emb, calendar_emb], dim=1)
        
        # Get conditioning parameters (gamma, beta for FiLM)
        gamma_beta = self.conditioning_net(combined)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        
        return gamma, beta
    
    
    def forward(
        self, 
        llm_embedding: torch.Tensor, 
        station_id: torch.Tensor, 
        calendar_features: torch.Tensor, 
        initial_sequence: torch.Tensor=None,
        sample: bool=True,
        mask: torch.Tensor=None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the decoder.

        Args:
            llm_embedding (torch.Tensor): LLM embedding tensor [batch_size, d_emb]
            station_id (torch.Tensor): Station IDs [batch_size]
            calendar_features (torch.Tensor): Calendar feature tensor [batch_size, calendar_feature_dim]
            initial_sequence (Optional[torch.Tensor], optional): Initial sequence for autoregressive generation. 
                Defaults to None.
            sample (bool, optional): Whether to sample from latent distribution during inference. 
                Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with model outputs including mean, logvar, mu, and logvar_z
        """
        # 1. Variational encoding
        mu, logvar_z = self.encode(llm_embedding)
        
        # Sample from latent space (use reparameterization if training)
        if self.training or sample:
            z = self.reparameterize(mu, logvar_z)
        else:
            z = mu
        
        # 2. Get conditioning signals
        gamma, beta = self.condition_with_metadata(z=z, station_id=station_id, calendar_features=calendar_features)
        
        # 3. Prepare decoder input sequence
        # Token embedding
        seq_embed = self.token_embedding(initial_sequence)
        
        # Apply conditioning (FiLM)
        gamma = gamma.unsqueeze(1).expand(-1, seq_embed.size(1), -1)  # [batch, seq_len, d_model]
        beta = beta.unsqueeze(1).expand(-1, seq_embed.size(1), -1)    # [batch, seq_len, d_model]
        seq_embed = gamma * seq_embed + beta
        
        # Apply positional encoding
        seq_embed = self.positional_encoding(seq_embed)
    
        # Pass through transformer. The src_key_padding_mask works for the initial_sequence -- the positions have missing values
        if not self.training and mask.numel() == mask.shape[1]:
            transformer_out = self.transformer.forward(src=seq_embed)
        else:
            transformer_out = self.transformer.forward(src=seq_embed, src_key_padding_mask=mask.squeeze(-1))
        
        # 6. Output with uncertainty
        mean = self.mean_output(transformer_out)
        logvar = self.logvar_output(transformer_out)
        
        # Apply activation to ensure non-negative predictions (optional)
        mean = F.softplus(mean)  # Uncomment for strictly positive outputs
        
        return {
            'mean': mean,
            'logvar': logvar,
            'mu': mu,
            'logvar_z': logvar_z
        }
    

    
    
    def generate_samples(
        self, 
        llm_embedding: torch.Tensor, 
        station_id: torch.Tensor, 
        calendar_features: torch.Tensor, 
        num_samples: int = 10,
        initial_sequence: torch.Tensor=None,
        mask: torch.Tensor=None,
        sample: bool=True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multiple samples for uncertainty estimation.

        Args:
            llm_embedding (torch.Tensor): LLM embedding tensor [batch_size, d_emb]
            station_id (torch.Tensor): Station IDs [batch_size]
            calendar_features (torch.Tensor): Calendar feature tensor [batch_size, calendar_feature_dim]
            num_samples (int, optional): Number of Monte Carlo samples. Defaults to 10.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with mean, standard deviation, and confidence intervals
        """
        
        batch_size = llm_embedding.size(0)
        all_samples = torch.zeros(
            num_samples, batch_size, self.seq_length, self.output_dim, 
            device=llm_embedding.device
        )
        
        for i in range(num_samples):
            # Forward pass with sampling enabled
            outputs = self.forward(
                llm_embedding=llm_embedding, 
                station_id=station_id, 
                calendar_features=calendar_features,
                sample=sample,  # Enable sampling from latent distribution
                initial_sequence=initial_sequence,
                mask=mask
            )
            
            # Store sample
            all_samples[i] = outputs['mean']
        
        # Calculate statistics
        mean_pred = all_samples.mean(dim=0)
        std_pred = all_samples.std(dim=0)
        
        # Confidence intervals (95%)
        lower_ci = mean_pred - 1.96 * std_pred
        upper_ci = mean_pred + 1.96 * std_pred
        
        # Ensure non-negative values for charging demand
        lower_ci = torch.clamp(lower_ci, min=0.0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'samples': all_samples
        }