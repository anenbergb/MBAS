def alpha_linear(epoch: int, max_epochs: int):
    return epoch / max_epochs


def alpha_stepwise(epoch: int, max_epochs: int, h: int):
    steps = epoch // h
    total_steps = max_epochs // h
    return steps / total_steps


def alpha_stepwise_warmup(
    epoch: int, max_epochs: int, h: int, warmup_epochs=250, max_alpha=0.75
):
    assert max_epochs > warmup_epochs
    p_epochs = max(epoch - warmup_epochs, 0)
    steps = p_epochs // h
    total_steps = (max_epochs - warmup_epochs) // h
    alpha = steps / total_steps
    return min(alpha, max_alpha)


def alpha_stepwise_warmup_scaled(
    epoch: int, max_epochs: int, h: int, warmup_epochs=250, max_alpha=0.75
):
    assert max_epochs > warmup_epochs
    assert max_alpha >= 0 and max_alpha <= 1, "alpha must be in [0, 1]"
    p_epochs = max(epoch - warmup_epochs, 0)
    steps = p_epochs // h
    total_steps = (max_epochs - warmup_epochs) // h
    alpha = max_alpha * (steps / total_steps)
    return alpha
