
"""Train CSDI using *already prepared* patient batches.

The optimizer, scheduler, random training masks, and diffusion loss are kept
from Matas5/CSDI/utils.py; normalization and validation masks come from the
shared preprocessing pipeline. The original code merely tracked the lowest
validation loss while saving the FINAL state at the end. We save final.pth
AND best.pth separately, so checkpoint choice is explicit.
"""
from pathlib import Path
import math

import torch
from torch.optim import Adam
from tqdm.auto import tqdm


def train(
    model,
    config,
    train_loader,
    valid_loader,
    output_dir,
    valid_epoch_interval=20,
    max_train_batches=None,
):
    """Return paths for final and optional best-validation state dictionaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    n_epochs = int(config["epochs"])
    if n_epochs < 1:
        raise ValueError("epochs must be positive")
    # SAME AS ORIGINAL: epochs at 75% and 90% trigger LR reductions.
    milestones = [int(0.75 * n_epochs), int(0.9 * n_epochs)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    max_steps = int(config["itr_per_epoch"])
    if max_train_batches is not None:
        max_steps = min(max_steps, max_train_batches)
    if max_steps < 1:
        raise ValueError("max_train_batches and itr_per_epoch must be positive")

    best_loss = math.inf
    best_path = output_dir / "best.pth"
    final_path = output_dir / "final.pth"

    for epoch in range(n_epochs):
        model.train()
        total_train, n_train = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Train {epoch+1}/{n_epochs}", leave=False):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            total_train += float(loss.item())
            n_train += 1
            if n_train >= max_steps:
                break
        if n_train == 0:
            raise RuntimeError("The training data loader was empty")
        scheduler.step()
        print(f"epoch {epoch+1}: train_loss={total_train/n_train:.6f}")

        if valid_loader is not None and (epoch + 1) % valid_epoch_interval == 0:
            model.eval()
            total_val, n_val = 0.0, 0
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc="Validate", leave=False):
                    # SAME AS ORIGINAL: validation loss averages ALL diffusion steps,
                    # which can be expensive even if the training epoch is short.
                    loss = model(batch, is_train=0)
                    total_val += float(loss.item())
                    n_val += 1
            if n_val == 0:
                raise RuntimeError("The validation data loader was empty")
            avg_val = total_val / n_val
            print(f"epoch {epoch+1}: valid_loss={avg_val:.6f}")
            if avg_val < best_loss:
                best_loss = avg_val
                # CHANGE vs original: actually save the best validation checkpoint.
                torch.save(model.state_dict(), best_path)

    # ORIGINAL BEHAVIOR available independently as final.pth.
    torch.save(model.state_dict(), final_path)
    return {"final": final_path, "best": best_path if best_path.exists() else None}
