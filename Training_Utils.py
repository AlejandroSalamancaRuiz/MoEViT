import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import math

def create_unseen_embds(model, id_imgs_loader, device):

    model.to(device)
    embd_cpu, labels_cpu = None, None
    start = True

    for images, labels in id_imgs_loader:
        images, labels = images.to(device), labels.to(device)
        logits, embeddings, loss = model(images, labels)

        if start:
            labels_cpu = labels.to('cpu')
            embd_cpu = embeddings.to('cpu')
            start = False
        else:
            embd_cpu = torch.cat((embd_cpu, embeddings.to('cpu')), dim=0)
            labels_cpu = torch.cat((labels_cpu, labels.to('cpu')))


    return embd_cpu, labels_cpu


def rank_acc(embeddings, labels):

    # Compute cosine similarity using efficient matrix operations
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarities = torch.mm(embeddings_norm, embeddings_norm.t())
    similarities.fill_diagonal_(0)  # Remove self-similarity
    
    # Sort and get indices of top matches
    sorted_indices = similarities.argsort(descending=True)

    rank_1_acc = 0.0
    rank_5_acc = 0.0

    for i in range(5):
        rank_5_acc += (labels[sorted_indices[:, i]] == labels).sum().item() 
        if i == 0:
            rank_1_acc = rank_5_acc
        
    return rank_1_acc / labels.shape[0], rank_5_acc / labels.shape[0]


def get_lr(it, warmup_steps, min_lr, max_lr, max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)