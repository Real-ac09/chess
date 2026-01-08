import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import glob
import os
import sys
from model import ChessNet

# --- CONFIG ---
DATA_DIR = "./processed_data"
BATCH_SIZE = 2048  # Large batch size for BF16 efficiency
LR = 0.001
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PREFIX = "chess_model_epoch_"

# --- LAZY DATASET ---
class ChessIterableDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Split files among workers to avoid duplication
        if worker_info is None:
            my_files = self.file_list
        else:
            per_worker = int(np.ceil(len(self.file_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_list))
            my_files = self.file_list[iter_start:iter_end]

        # Shuffle file order
        np.random.shuffle(my_files)
        
        for f in my_files:
            try:
                with np.load(f) as data:
                    inputs = data['inputs']
                    policies = data['policies']
                    values = data['values']
                    
                # Shuffle positions within the file
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
                
                for i in indices:
                    yield {
                        'input': inputs[i],   # Keeps original dtype (float16 from disk)
                        'policy': policies[i],
                        'value': values[i]
                    }
            except Exception as e:
                print(f"Skipping bad file {f}: {e}")

# --- TRAINING LOOP ---
def train():
    print(f"System: {DEVICE.upper()}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for extra speed on Ampere (3090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Initialize Model
    model = ChessNet().to(DEVICE)
    
    # 2. COMPILE MODEL (The new addition)
    # This fuses layers for faster execution. 
    # Expect a delay at the start of training while it compiles.
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()... (Wait ~60s)")
        model = torch.compile(model)
    else:
        print("Warning: torch.compile not found. Update PyTorch for extra speed.")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') 

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # Resume Logic
    start_epoch = 0
    existing_ckpts = glob.glob(f"{CHECKPOINT_PREFIX}*.pt")
    if existing_ckpts:
        try:
            latest_ckpt = max(existing_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            print(f"Resuming from {latest_ckpt}")
            # Note: We load weights BEFORE compile usually, but loading into compiled model works in 2.0+
            # If resume fails, load into raw model first, then compile.
            state_dict = torch.load(latest_ckpt)
            # Handle potential prefix issues if loading compiled weights into uncompiled model or vice versa
            # (Usually handled automatically by torch, but simple load is standard)
            model.load_state_dict(state_dict) 
            start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0])
        except Exception as e: 
            print(f"Could not load checkpoint: {e}")

    # Data Setup
    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    if not files:
        print("No data found.")
        return
    print(f"Found {len(files)} data files. Using Streaming Mode with BF16 AMP.")

    # Main Loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        dataset = ChessIterableDataset(files)
        # 4 workers is usually optimal for feeding a fast GPU
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) 

        total_p_loss = 0
        total_v_loss = 0
        batch_count = 0

        print(f"--- Epoch {epoch+1} Start ---")
        
        for i, batch in enumerate(loader):
            inputs = batch['input'].to(DEVICE, non_blocking=True) 
            target_policy = batch['policy'].long().to(DEVICE, non_blocking=True)
            target_value = batch['value'].float().to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            
            # --- MIXED PRECISION FORWARD PASS ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_policy, pred_value = model(inputs)
                l_p = ce_loss(pred_policy, target_policy)
                l_v = mse_loss(pred_value, target_value)
                loss = l_p + l_v
            
            # --- SCALED BACKWARD PASS ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_p_loss += l_p.item()
            total_v_loss += l_v.item()
            batch_count += 1
            
            if i % 100 == 0:
                print(f"Batch {i} | P-Loss: {l_p.item():.3f} | V-Loss: {l_v.item():.3f}")

        if batch_count > 0:
            avg_loss = (total_p_loss+total_v_loss)/batch_count
            print(f"--> EPOCH {epoch+1} DONE. Avg Loss: {avg_loss:.4f}")
            # Save the underlying model (unwrap from compile if needed, though state_dict handles it)
            torch.save(model.state_dict(), f"{CHECKPOINT_PREFIX}{epoch+1}.pt")
        else:
            print("Warning: Epoch finished with 0 batches?")

if __name__ == "__main__":
    train()
