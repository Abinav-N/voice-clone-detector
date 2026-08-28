"""
Lightweight 2D CNN over the (3, n_mfcc, T) MFCC/delta/delta2 stack.
Small enough to train on CPU or a free-tier GPU in well under an hour.
"""
import torch
import torch.nn as nn


class VoiceCloneCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(in_channels, 16),   # -> (16, n_mfcc/2, T/2)
            conv_block(16, 32),            # -> (32, n_mfcc/4, T/4)
            conv_block(32, 64),            # -> (64, n_mfcc/8, T/8)
            conv_block(64, 128),           # -> (128, n_mfcc/16, T/16)
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)   # raw logits, shape (B, 2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = VoiceCloneCNN()
    dummy = torch.randn(4, 3, 40, 300)  # batch of 4
    out = m(dummy)
    print("output shape:", out.shape)
    print("trainable params:", count_parameters(m))
