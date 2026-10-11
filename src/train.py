import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data import SEP_ID, SEQ_LEN, VOCAB_SIZE, make_batch, make_eval_pair
from model import TinyTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@torch.no_grad()
def generate(model, x, seq_len, device):
    model.eval()
    sep = torch.full((x.shape[0], 1), SEP_ID, device=device, dtype=x.dtype)
    seq = torch.cat([x, sep], dim=1)

    for _ in range(seq_len):
        logits, _ = model(seq)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)

    model.train()
    return seq[:, -seq_len:]


@torch.no_grad()
def exact_match_accuracy(model, mode, device, seq_len=SEQ_LEN, batch_size=512):
    x, y_true = make_eval_pair(batch_size, mode, device, seq_len)
    y_pred = generate(model, x, seq_len, device)
    return (y_pred == y_true).all(dim=1).float().mean().item()


def train(mode, num_steps=2000, batch_size=128, lr=1e-3, log_every=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = TinyTransformer(VOCAB_SIZE, max_len=2 * SEQ_LEN, d_model=64, d_ff=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"--- training mode={mode!r} on {device} ---")
    for step in range(1, num_steps + 1):
        inputs, targets, loss_mask, _, _ = make_batch(batch_size, mode, device)

        logits, _ = model(inputs)
        loss = F.cross_entropy(logits[loss_mask], targets[loss_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == num_steps:
            acc = exact_match_accuracy(model, mode, device)
            print(f"step {step:5d} | loss {loss.item():.4f} | exact-match acc {acc:.3f}")

    final_acc = exact_match_accuracy(model, mode, device, batch_size=2000)
    print(f"--- final held-out exact-match accuracy ({mode}): {final_acc:.4f} ---")

    DATA_DIR.mkdir(exist_ok=True)
    ckpt_path = DATA_DIR / f"{mode}_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

    return model, final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["copy", "reverse", "both"], default="both")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    for m in (["copy", "reverse"] if args.mode == "both" else [args.mode]):
        train(m, num_steps=args.steps)
