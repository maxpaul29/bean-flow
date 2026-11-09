import torch.nn as nn

from src.data import set_seed

# 8 Linear layers: 7 hidden (with activation) + 1 output layer returning raw logits
# no softmax at the end because nn.CrossEntropyLoss applies log_softmax internally
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, activation='sigmoid'):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer6 = nn.Linear(hidden_dim, hidden_dim)
        self.layer7 = nn.Linear(hidden_dim, hidden_dim)
        self.layer8 = nn.Linear(hidden_dim, out_features)

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation function. Choose 'sigmoid' or 'relu'.")

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        x = self.activation(self.layer5(x))
        x = self.activation(self.layer6(x))
        x = self.activation(self.layer7(x))
        return self.layer8(x)

# xavier-uniform weights and zero biases
# xavier keeps activation variance roughly constant for symmetric activations like sigmoid
# the same init is used for ReLU on purpose, so the activation is the only difference between runs
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# fresh model per run, seeded before init so the weights are reproducible
def build_mlp(activation, seed, in_features=16, hidden_dim=64, out_features=7):
    set_seed(seed)
    model = MLP(in_features, hidden_dim, out_features, activation)
    model.apply(init_weights)
    return model
