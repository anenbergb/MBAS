def alpha_linear(epoch: int, max_epochs: int):
    return 1 - epoch / max_epochs


def alpha_stepwise(epoch: int, max_epochs: int, h: int):
    steps = epoch // h
    total_steps = max_epochs // h
    return 1 - steps / total_steps
