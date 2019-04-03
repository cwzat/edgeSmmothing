def mixup(mask, gen, ori):
    return mask * gen + (1 - mask) * ori