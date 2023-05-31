
import torch
import torch.nn as nn

def conv(ni, nf, kernel_size=3, stride=1, padding='same', **kwargs):
    _conv = nn.Conv1d(ni, nf, kernel_size=kernel_size,stride=stride,padding=padding, **kwargs)
    nn.init.kaiming_normal_(_conv.weight)
    return _conv

def get_locnet(channels=1, output_dim=64):
    # Spatial transformer localization-network
    locnet = nn.Sequential(
        conv(channels, 128, kernel_size=8),
        nn.BatchNorm1d(128),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU(True),
        conv(128, 64, kernel_size=5),
        nn.BatchNorm1d(64),
        nn.MaxPool1d(3, stride=3),
        nn.ReLU(True),
        conv(64, 64, kernel_size=3),
        nn.BatchNorm1d(64),
        nn.MaxPool1d(3, stride=2),
        nn.ReLU(True),
        # GAP (when size=1) -
        # Note: While GAP allow the model size to remain fix w.r.t input length,
        # Temporal information is lost by the GAP operator.
        nn.AdaptiveAvgPool1d(output_size=(1)),
        nn.Flatten(),
        nn.Linear(64, output_dim),
        nn.ReLU(True),
    )
    return locnet




class EnsembleModel(nn.Module):   
    def __init__(self, n_ensemble=5, channels=1, output_dim=64):
        super().__init__()

        self.module_list = nn.ModuleList([get_locnet(channels, output_dim) for _ in range(n_ensemble)])
        self.n_ensemble = n_ensemble
        self.gap =  nn.AdaptiveAvgPool1d(output_size=(1))
    def forward(self, x):
        preds = []
        for i in range(self.n_ensemble):
            preds.append(self.module_list[i](x))
        x = torch.stack(preds, dim=-1)
        #print("cat.shape",x.shape)
        out = self.gap(x)
        out = torch.squeeze(out, dim=-1)
        #print("out.shape",out.shape)
        return out


def get_CNN_ensemble(n_ensemble=10, channels=1, output_dim=64):

    ensemble_model = EnsembleModel(n_ensemble, channels, output_dim)
    return ensemble_model