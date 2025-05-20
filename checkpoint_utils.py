import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoint.pth", store_checkpoint_for_every_epoch=False):
    """Save model and optimizer state to a checkpoint file."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    if store_checkpoint_for_every_epoch:
        # If the checkpoint has to be stored for every epoch to the name of the checkpoint at the end will be added _ep{epoch}
        # This is implemented since if you store all checkpoints to the same filename the next epoch will override the results of the previous epoch
        checkpoint_path = checkpoint_path[:checkpoint_path.rfind('.')] + f"_ep{epoch}" + checkpoint_path[checkpoint_path.rfind('.')+1:]
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from a checkpoint file if it exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: Resuming from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0, None  # Start from epoch 0 if no checkpoint is found