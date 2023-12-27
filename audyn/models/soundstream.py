from .rvqvae import RVQVAE

__all__ = ["SoundStream"]


class SoundStream(RVQVAE):
    """Sound stream using residual vector quantizer.

    Args:
        encoder (nn.Module): Encoder which returns latent feature of
            shape (batch_size, embedding_dim, *).
        decoder (nn.Module): Decoder which takes latent feature of
            shape (batch_size, embedding_dim, *).
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimension.
        num_layers (int): Number of layers of RVQ.

    """
