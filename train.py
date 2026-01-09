import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import glob
import os
import sys
from tqdm import tqdm
from model import ChessNet

# --- CONFIG ---
DATA_DIR = "./processed_data"
BATCH_SIZE = 2048
LR = 0.02 # Pure Muon LR
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PREFIX = "chess_model_"
SAVE_INTERVAL = 5000 
ESTIMATED_BATCHES_PER_EPOCH = 83500

# --- PURE MUON OPTIMIZER ---
class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                # Momentum Buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # --- THE FIX: Only run Newton-Schulz on 2D+ tensors ---
                if p.ndim >= 2:
                    # Orthogonalize Weights
                    update = self._newton_schulz(g, ns_steps)
                else:
                    # Standard SGD update for Biases (1D)
                    # We just use the momentum-accelerated gradient directly
                    update = g

                # Apply Update
                p.data.add_(update, alpha=-lr)

        return loss

    def _newton_schulz(self, G, steps=5, eps=1e-7):
        original_shape = G.shape
        if len(original_shape) > 2:
            G = G.view(G.size(0), -1) 
        
        X = G
        norm = X.norm() + eps
        X = X / norm 

        for _ in range(steps):
            A = X @ X.T
            B = 1.5 * X - 0.5 * A @ X
            X = B
        
        return X.view(original_shape)

# --- LAZY DATASET ---
class ChessIterableDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_files = self.file_list
        else:
            per_worker = int(np.ceil(len(self.file_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_list))
            my_files = self.file_list[iter_start:iter_end]

        np.random.shuffle(my_files)
        for f in my_files:
            try:
                with np.load(f) as data:
                    inputs = data['inputs']
                    policies = data['policies']
                    values = data['values']
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
                for i in indices:
                    yield {'input': inputs[i], 'policy': policies[i], 'value': values[i]}
            except: pass

# --- TRAINING LOOP ---
def train():
    print(f"System: {DEVICE.upper()}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = ChessNet().to(DEVICE)
    if hasattr(torch, 'compile'):
        print("Compiling model... (Wait ~60s)")
        model = torch.compile(model)

    print("Using Pure Muon Optimizer")
    optimizer = Muon(model.parameters(), lr=LR)
    
    scaler = torch.amp.GradScaler('cuda') 
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    start_epoch = 0
    existing_ckpts = glob.glob(f"{CHECKPOINT_PREFIX}*.pt")
    if existing_ckpts:
        try:
            latest_ckpt = max(existing_ckpts, key=os.path.getctime)
            print(f"Resuming weights from {latest_ckpt}")
            state_dict = torch.load(latest_ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("_orig_mod.", "")] = v
            model.load_state_dict(new_state_dict)
            if "epoch_" in latest_ckpt:
                 parts = latest_ckpt.split('epoch_')[1].split('_')
                 start_epoch = int(parts[0])
        except Exception as e: 
            print(f"Could not parse checkpoint: {e}")

    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    if not files: return

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        dataset = ChessIterableDataset(files)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) 
        
        pbar = tqdm(loader, total=ESTIMATED_BATCHES_PER_EPOCH, desc=f"Epoch {epoch+1}")
        
        total_p_loss = 0
        total_v_loss = 0
        batch_count = 0

        for batch in pbar:
            inputs = batch['input'].to(DEVICE, non_blocking=True)
            target_policy = batch['policy'].long().to(DEVICE, non_blocking=True)
            target_value = batch['value'].float().to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_policy, pred_value = model(inputs)
                l_p = ce_loss(pred_policy, target_policy)
                l_v = mse_loss(pred_value, target_value)
                loss = l_p + l_v
            
            scaler.scale(loss).backward()
            
            # Explicit Unscale for Muon correct math
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scaler.update()
            
            total_p_loss += l_p.item()
            total_v_loss += l_v.item()
            batch_count += 1
            
            pbar.set_postfix({'P-Loss': f"{l_p.item():.3f}", 'V-Loss': f"{l_v.item():.3f}"})

            if batch_count % SAVE_INTERVAL == 0:
                savename = f"{CHECKPOINT_PREFIX}epoch_{epoch+1}_batch_{batch_count}.pt"
                torch.save(model.state_dict(), savename)

        avg_loss = (total_p_loss + total_v_loss) / max(1, batch_count)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"{CHECKPOINT_PREFIX}epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
