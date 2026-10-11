import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import SEQ_LEN, VOCAB_SIZE, make_batch, token_label
from model import TinyTransformer
from train import exact_match_accuracy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_model(mode, device):
    model = TinyTransformer(VOCAB_SIZE, max_len=2 * SEQ_LEN, d_model=64, d_ff=128).to(device)
    model.load_state_dict(torch.load(DATA_DIR / f"{mode}_model.pt", map_location=device))
    model.eval()
    return model


def plot_attention(attn, tokens, mode, example_idx):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(attn.numpy(), cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    ax.set_xlabel("attending to (key position)")
    ax.set_ylabel("attending from (query position)")
    ax.set_title(f"{mode} — example {example_idx}: attention weights")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("attention weight")

    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / f"{mode}_example{example_idx}_attention.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


@torch.no_grad()
def show_examples(model, mode, device, num_examples=4):
    inputs, _, _, x, y = make_batch(num_examples, mode, device)
    _, attn_weights = model(inputs)

    for i in range(num_examples):
        tokens = [token_label(t.item()) for t in inputs[i]]
        path = plot_attention(attn_weights[i].cpu(), tokens, mode, i)
        print(f"  x={x[i].tolist()} y={y[i].tolist()} -> saved {path.name}")


def main(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(mode, device)

    acc = exact_match_accuracy(model, mode, device, batch_size=2000)
    print(f"[{mode}] held-out exact-match accuracy: {acc:.4f}")

    print(f"[{mode}] attention heatmaps for a few examples:")
    show_examples(model, mode, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["copy", "reverse", "both"], default="both")
    args = parser.parse_args()

    for m in (["copy", "reverse"] if args.mode == "both" else [args.mode]):
        main(m)
