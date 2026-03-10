from __future__ import annotations

import torch

from CorridorKeyModule.core.model_transformer import DecoderHead


def test_decoder_head_accepts_non_contiguous_intermediates():
    head = DecoderHead(feature_channels=[8, 16, 32, 64], embedding_dim=16, output_dim=1)
    head.eval()

    features = [
        torch.randn(2, 8, 32, 32),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]

    with torch.inference_mode():
        output = head(features)

    assert output.shape == (2, 1, 32, 32)
