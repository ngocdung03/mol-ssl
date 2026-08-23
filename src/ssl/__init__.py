"""Semi-supervised methods behind one interface, so ablations are config flips not code forks.

Each module exposes:
    build(cfg) -> object with .loss(batch, model) -> torch.Tensor
NOT YET IMPLEMENTED — M3. Interface fixed here so M1/M2 code can be written against it.
"""
