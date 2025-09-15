import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool



class TargetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):  # Binary classification
        super(TargetModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        return conv1,conv2


class HitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(HitModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

         return conv1,conv2

class ADMETModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=5):  # ADMET properties
        super(ADMETModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

           return conv1,conv2
