import torch

def p_loss(target, weighted_pixels, p=1, reduction='sum'):
    b, t, h, w, c = weighted_pixels.shape
    assert target.shape == (b, t, h, w, c)

    loss = torch.abs(weighted_pixels - target)
    if p < 1:
        loss = torch.pow(loss + 1e-10, p)
    else:
        loss = torch.pow(loss, p)
    if reduction == 'sum':
        loss = loss.sum(dim=(4, 3, 2, 1))
    elif reduction == 'mean':
        loss = loss.mean(dim=(4, 3, 2, 1))

    assert loss.shape == (b,)
    
    return loss
