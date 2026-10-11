import torch

DIGIT_VOCAB = 10
SEP_ID = DIGIT_VOCAB
VOCAB_SIZE = DIGIT_VOCAB + 1
SEQ_LEN = 5


def _make_targets(x, mode):
    if mode == "copy":
        return x.clone()
    if mode == "reverse":
        return x.flip(dims=[1])
    raise ValueError(f"unknown mode: {mode!r}")


def make_batch(batch_size, mode, device, seq_len=SEQ_LEN):
    # sequence is [x0..x4, SEP, y0..y4]; inputs/targets are that shifted by one
    # token, so the model predicts each next token given everything before it
    x = torch.randint(0, DIGIT_VOCAB, (batch_size, seq_len), device=device)
    y = _make_targets(x, mode)
    sep = torch.full((batch_size, 1), SEP_ID, device=device, dtype=x.dtype)

    full = torch.cat([x, sep, y], dim=1)
    inputs = full[:, :-1]
    targets = full[:, 1:]

    # only score predictions that fall on the y half
    loss_mask = torch.zeros_like(targets, dtype=torch.bool)
    loss_mask[:, seq_len:] = True

    return inputs, targets, loss_mask, x, y


def make_eval_pair(batch_size, mode, device, seq_len=SEQ_LEN):
    x = torch.randint(0, DIGIT_VOCAB, (batch_size, seq_len), device=device)
    return x, _make_targets(x, mode)


def token_label(token_id):
    return "SEP" if token_id == SEP_ID else str(token_id)
