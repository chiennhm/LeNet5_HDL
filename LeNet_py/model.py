import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),    # (1,28,28) -> (6,24,24)
            nn.ReLU(),
            nn.AvgPool2d(2),                   # -> (6,12,12)

            nn.Conv2d(6, 16, kernel_size=5),   # -> (16,8,8)
            nn.ReLU(),
            nn.AvgPool2d(2),                   # -> (16,4,4)

            nn.Conv2d(16, 120, kernel_size=4), # -> (120,1,1)
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model):
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        print(f"  {name}: {n}")
        total += n
    print(f"  Total: {total}")
    return total
