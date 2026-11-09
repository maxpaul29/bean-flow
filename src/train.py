import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import SEED, load_dry_bean_splits, make_loader, set_seed
from src.diagnostics import layer_grad_norms
from src.model import build_mlp


def train(activation, clip_norm=None, seed=SEED, num_steps=2000, lr=0.01,
          batch_size=32, record_every=10, train_set=None):
    # the gradient norms are recorded on training data only
    if train_set is None:
        train_set, _ = load_dry_bean_splits()

    model = build_mlp(activation, seed)
    loader = make_loader(train_set, batch_size, seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    max_norm = clip_norm if clip_norm is not None else float("inf")  # inf: measure only, never scale

    grad_matrix, pre_clip_norms, losses = [], [], []
    step = 0
    model.train()
    while step < num_steps:
        for xb, yb in loader:
            if step >= num_steps:
                break
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            if step % record_every == 0:
                grad_matrix.append(layer_grad_norms(model))
                pre_clip_norms.append(pre_clip.item())
            losses.append(loss.item())

            optimizer.step()
            step += 1

    return {
        "grad_matrix": np.array(grad_matrix),
        "pre_clip_norms": np.array(pre_clip_norms),
        "losses": np.array(losses),
        "model": model,
    }

def evaluate(model, dataset, batch_size=1024):
    # Accuracy on a dataset (train_set or test_set) in eval mode without gradient tracking
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    return correct / len(dataset)

def overfit_one_batch(activation, seed=SEED, batch_size=32, num_steps=500, lr=0.05):
    # sanity check: a working training loop must drive the loss on one batch close to 0
    train_set, _ = load_dry_bean_splits()
    xb, yb = next(iter(make_loader(train_set, batch_size, seed)))

    model = build_mlp(activation, seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    return loss.item()
