from math import fmod
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from SparseMoE import SparseMoE
from Attention import MQGAttention, precompute_theta_pos_frequencies
import inspect

@dataclass
class MoEViTConfig:

    img_dim: int = 112
    patch_size: int = 7

    block_size = int((img_dim / patch_size)**2)

    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 6

    n_embd: int = 768
    dropout: float = 0.0

    num_experts: int = 6
    active_experts: int = 2
    norm_eps: float = 1e-6
    
    img_embd: int = 768
    num_classes: int = 10000

    s: int = 64
    m: float = 0.35

    device: str = 'cpu'



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.n_embd
        self.head_dim = config.n_embd // config.n_heads

        self.attention = MQGAttention(config)
        self.smoe = SparseMoE(config)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(self.dim, eps=config.norm_eps)
        # Normalization BEFORE the feed forward block
        self.smoe_norm = RMSNorm(self.dim, eps=config.norm_eps)
    
    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), freqs_complex)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.smoe(self.smoe_norm(h))
        return out


class MoEViT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        assert  self.config.n_heads % self.config.n_kv_heads == 0, "the number of queries should be a multiple of k's and v's"

        self.transformer = nn.ModuleDict(dict(
            te = nn.Conv2d(3, config.n_embd, config.patch_size, stride=config.patch_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)]),
            norm_l = RMSNorm(config.n_embd, eps=config.norm_eps),
        ))

        self.embedding_layer = nn.Linear(config.n_embd, config.img_embd)

        self.clf_head = nn.Linear(config.img_embd, config.num_classes, bias=False)
        # block_size+1 toaccount for the CLF token!
        self.freqs_complex = precompute_theta_pos_frequencies(self.config.n_embd // self.config.n_heads, self.config.block_size+1, device=self.config.device)

        # init params
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # just for projections of expert modules (GPT2)
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def get_num_params(self):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        num_exp = self.config.active_experts
        total_params, active_params = 0,0

        for key in param_dict:
            total_params += param_dict[key].numel()

            name = key.split('.')
            if name[1] == 'h' and name[3] == 'smoe' and name[4] == 'experts':
                if name[5] == '0':
                    active_params += param_dict[key].numel() * num_exp

            else: active_params += param_dict[key].numel()

        return total_params, active_params


    def forward(self, img_batch, targets=None):

        tok_emb = self.transformer.te(img_batch) 

        b, ch, w, h = tok_emb.size()
        t = w*h

        tok_emb = tok_emb.view(b, ch, t)
        tok_emb = torch.swapaxes(tok_emb, 1,2)
        clf_tok = torch.ones((b,1,tok_emb.shape[2]), device=self.config.device)
        tok_emb = torch.cat((clf_tok, tok_emb), dim=1) 
   
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x, self.freqs_complex)

        x = self.transformer.norm_l(x)

        x = x[:,0,:]

        img_embds = self.embedding_layer(x)
        logits = self.clf_head(img_embds)

        if targets is not None:
            loss = cos_face_loss(img_embds, self.clf_head.weight, self.config.s, self.config.m, targets, self.config.device)
        else: loss=None
        
        return logits, img_embds, loss
    

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


def cos_face_loss(X, W, s, m, y, device):

    # Ensure normalization
    W_norm = torch.norm(W, dim=1, keepdim=True)
    X_norm = torch.norm(X, dim=1, keepdim=True)

    # Normalize weights and inputs
    W_n = W / W_norm
    X_n = X / X_norm

    # Compute cosine similarity
    cos_vals = (W_n @ X_n.T).T

    # Create masks
    mask = F.one_hot(y, num_classes=W.shape[0]).to(device).float()
    reversed_mask = 1.0 - mask

    # Exponential terms for softmax
    exp_logits = torch.exp(s * cos_vals) * reversed_mask
    exp_logits_m = torch.exp(s * (cos_vals - m)) * mask

    numerator = torch.sum(exp_logits_m, dim=1)
    denominator = torch.sum(exp_logits, dim=1) + numerator

    # Final loss calculation
    loss = -torch.log(numerator / denominator)

    return torch.mean(loss)
    

