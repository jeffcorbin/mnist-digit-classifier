import torch
import torch.nn as nn

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

def load_model(path="app/model.pt"):
    model = DigitClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model
