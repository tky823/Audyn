import torch

from audyn.modules import PoincareEmbedding
from audyn.modules.manifold import NegativeSamplingPoincareEmbedding
from audyn.modules.negative_sampling import NegativeSamplingModel


def test_negative_sampling_model() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_neg_samples = 2
    num_embeddings, embedding_dim = 10, 2

    anchor = torch.randint(0, num_embeddings, (batch_size,), dtype=torch.long)
    positive = torch.randint(0, num_embeddings, (batch_size,), dtype=torch.long)
    negative = torch.randint(0, num_embeddings, (batch_size, num_neg_samples), dtype=torch.long)

    model = NegativeSamplingPoincareEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )

    anchor, positive, negative = model(anchor, positive, negative)

    assert anchor.size() == (batch_size, embedding_dim)
    assert positive.size() == (batch_size, embedding_dim)
    assert negative.size() == (batch_size, num_neg_samples, embedding_dim)

    embedding = PoincareEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    )
    model = NegativeSamplingModel(embedding)

    anchor = torch.randint(0, num_embeddings, (batch_size,), dtype=torch.long)
    positive = torch.randint(0, num_embeddings, (batch_size,), dtype=torch.long)
    negative = torch.randint(0, num_embeddings, (batch_size, num_neg_samples), dtype=torch.long)

    anchor, positive, negative = model(anchor, positive, negative)

    assert anchor.size() == (batch_size, embedding_dim)
    assert positive.size() == (batch_size, embedding_dim)
    assert negative.size() == (batch_size, num_neg_samples, embedding_dim)
