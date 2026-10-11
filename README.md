# transformer-copy-reverse

A transformer built from scratch (no `nn.Transformer`, no `nn.MultiheadAttention`) trained to copy or reverse short digit sequences.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/train.py --mode both       # trains copy and reverse, saves data/*.pt
python src/evaluate.py --mode both    # prints accuracy, saves figures/*.png
```

## Files

- `src/model.py`: the model. One attention layer, one head, written out by hand so every step (Q/K/V projections, scaling, causal mask, softmax, weighted sum) is visible.
- `src/data.py`: builds `[x0..x4, SEP, y0..y4]` sequences and the mask that keeps loss off the x half.
- `src/train.py`: training loop, plus autoregressive generation for a real (non-teacher-forced) accuracy check.
- `src/evaluate.py`: loads a checkpoint, reports accuracy, saves attention heatmaps.

## Task

Everything is one flat sequence: `[x0 x1 x2 x3 x4 SEP y0 y1 y2 y3 y4]`. The model does next-token prediction across the whole thing, and the loss is masked so it's only graded on the y half.

## Attention heatmaps

Training is teacher-forced, so the easiest thing a single-head model can learn is a fixed positional offset:

- Copy: `y_i = x_i`, which sits exactly 5 positions ahead of it, so the model learns to attend 5 steps back every time. Shows up as a diagonal stripe.
- Reverse: `y_i = x_{4-i}`, so query position plus key position stays constant instead of their difference. Shows up as an anti-diagonal.

Saved in `figures/*_attention.png`, with axes labeled by the actual tokens.

## Results

Both modes hit 100% exact-match accuracy (whole sequence correct, not per-token) after 2000 steps.
