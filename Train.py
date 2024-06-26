import torch
import torch.nn.functional as F
import os
from Training_Utils import rank_acc, create_unseen_embds, get_lr
import argparse
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from MoEViT import MoEViT, MoEViTConfig
import time
import sys
import config
from Data_Utils import CustomDataLoader

# -----------------------------------------------------------------------------
# simple launch:
# python train.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train.py
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                               DDP configuration
# -----------------------------------------------------------------------------

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"


# -----------------------------------------------------------------------------
#                  Reproducibility and matmul optimization
# -----------------------------------------------------------------------------

## Seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Faster Matrix multiplication
torch.set_float32_matmul_precision('high')


# -----------------------------------------------------------------------------
#                              Model configuration
# -----------------------------------------------------------------------------

# Model instantiation
model_config = MoEViTConfig(device=device)
model = MoEViT(model_config)
model.to(device)
# Use torch compile for faster training. Or not...
#model = torch.compile(model)

if master_process:
    print(' ------------- Model  -------------')
    print(' ---------- -------------- ----------')
    print(f"Img dimension: {model_config.img_dim}")
    print(f"Patch size: {model_config.patch_size}")
    print(f"Block Size (Tokens): {model_config.block_size}")
    print(f"Num Encoder layers: {model_config.n_layers}")
    print(f"Embd dim: {model_config.n_embd}")
    print(f"Num heads x encoder: {model_config.n_heads}")
    print(f"Num key_values x encoder: {model_config.n_kv_heads}")
    print(f"Num Experts: {model_config.num_experts}")
    print(f"Num Active Experts: {model_config.active_experts}")
    print(f"Final Embd dim: {model_config.n_embd}")
    print(f"Num Classes: {model_config.num_classes}")
    print(f"Dropout: {model_config.dropout}")
    print(f"Norm eps: {model_config.norm_eps}")
    print(f"s and m for cos loss: {model_config.s} , {model_config.m}")
    total_params, active_params = model.get_num_params()
    total_params_f = "{:,}".format(total_params)
    active_params_f = "{:,}".format(active_params)
    print(f"Total num parameters: {total_params_f}")
    print(f"Active num parameters: {active_params_f}")
    print(' ---------- -------------- ----------')
    print(' ---------- -------------- ----------')


if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


# -----------------------------------------------------------------------------
#                             Training configuration
# -----------------------------------------------------------------------------

eval_interval = config.eval_interval
val_steps = config.val_steps

# Optimization hyperparameters following GPT3 paper
max_lr = config.max_lr
min_lr = config.min_lr
warmup_steps = config.warmup_steps
max_steps = config.max_steps
weight_decay = config.weight_decay

# Batch Size in tokens following GPT3 paper specs should be 0.5M
total_batch_size = config.total_batch_size
B = config.B
T = model_config.block_size
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

# Create Data Loaders
train_loader = CustomDataLoader(B, os.path.join(config.shards_folder, 'train'), process_rank=ddp_rank, num_processes=ddp_world_size)
val_loader = CustomDataLoader(B, os.path.join(config.shards_folder, 'validation'), process_rank=ddp_rank, num_processes=ddp_world_size)
id_imgs_loader = CustomDataLoader(B, os.path.join(config.shards_folder, 'identification_imgs'), process_rank=ddp_rank, num_processes=ddp_world_size)


if master_process:
    print(' ---------- Data Loaders ----------')
    for name, loader in [('Train',train_loader), ('Val', val_loader), ('Identification', id_imgs_loader)]:
        print(f"{name} -- shards in data loader: {len(loader.shards)} , batch_size: {loader.B}")
    print(' ---------- -------------- ----------')
    print(' ---------- -------------- ----------')
    print(' ---------- Configurations ----------')
    print(f"Tokens per image: {T}")
    print(f"Total desired batch size in images: {total_batch_size / T}")
    print(f"Total desired batch size in tokens: {total_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Maximum number of steps: {max_steps}")
    print(f"Validation steps: {val_steps}")
    print(f"Validation interval: {val_steps} steps")
    print(' ---------- -------------- ----------')
    print(' ---------- -------------- ----------')


# optimizer
print(' ------------ Optimizer ------------')
optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type, master_process=master_process)
print(' ---------- -------------- ----------')
print(' ---------- -------------- ----------')

# create the log directory 
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: 
    pass

# -----------------------------------------------------------------------------
#                             Training Loop
# -----------------------------------------------------------------------------

for step in range(max_steps):
    last_step = (step == max_steps - 1)

    model.train()
    optimizer.zero_grad()
    loss_accum, loss_accum1 = 0.0, 0.0
    t0 = time.time()


    for micro_step in range(grad_accum_steps):

        images, labels = train_loader.next_batch()
        # Move the images and labels to the device
        images = images.to(device)
        labels = labels.to(device)

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
    
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, embeddings, loss = model(images, labels)

        # Sanity check
        #loss_softm = F.cross_entropy(logits, labels) / grad_accum_steps 
        #loss_accum_sftm += loss_softm.detach()

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

        
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step, warmup_steps, min_lr, max_lr, max_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work

    t1 = time.time()
    dt = t1 - t0 # time difference in seconds

    images_processed = train_loader.B * grad_accum_steps * ddp_world_size
    images_per_sec = images_processed / dt

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | imgs/sec: {images_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")


    # -----------------------------------------------------------------------------
    #                             Evaluation Loop
    # -----------------------------------------------------------------------------
    if step % eval_interval == 0 or last_step:
        model.eval()
        rank_1 , rank_5 = 0.0, 0.0

        with torch.no_grad():
            val_loss_accum = 0.0
            embd_cpu, labels_cpu = create_unseen_embds(model, id_imgs_loader, config.id_eval_steps, device)
            rank_1 , rank_5  = rank_acc(embd_cpu, labels_cpu)

            for _ in range(val_steps):

                images, labels = val_loader.next_batch()
                # Move the images and labels to the device
                images = images.to(device)
                labels = labels.to(device)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, embeddings, loss = model(images, labels)

                loss = loss / val_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}, Rank-1-acc: {rank_1:.2f} , Rank-5-acc: {rank_5:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f} R1 {rank_1:.2f} R5 {rank_5:.2f} \n")

            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_path)


if ddp:
    destroy_process_group()



