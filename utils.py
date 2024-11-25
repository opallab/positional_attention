import torch

def get_loss(criterion, out, y, num_additional_node, n, target):
    if target == 'path':
        return criterion(torch.triu(out[:,:n,:], 1), torch.triu(y[:,:n,:], 1))
    else:
        if num_additional_node > 0:
            return criterion(out[:, :-num_additional_node], y[:, :-num_additional_node])
        else:
            return criterion(out, y)

def get_nlp_loss(criterion, out, y, num_additional_node, n, target):
        return criterion(out[:, :1], y[:, :1])

def get_accuracy(out, y, num_additional_node, n):
    """
    Return the accuracy of the model output `out` given the target `y`. If `target`
    is not 'median', the accuracy is calculated by rounding `out` to the nearest integer
    and comparing it with `y`. If `target` is 'median', the accuracy is calculated the same 
    way but only for odd positions (i.e. 1st, 3rd, 5th, etc.) in the sequence.
    """
    output, tgt = None, None
    if num_additional_node > 0:
        output = out[:, :-num_additional_node]
        tgt = y[:, :-num_additional_node]
    else:
        output = out
        tgt = y
    num_equal = ((output == tgt).sum(dim=1) == n).sum().item()
    return num_equal / out.size(0)

def get_close_accuracy(out, y, num_additional_node, n, rtol=0.1, atol=0.1):
    """
    Return the accuracy of the model output `out` given the target `y` allowing a relative 
    error of `rtol` and an absolute error of `atol`.
    Parameters:
        - rtol (float, default=0.01): relative tolerance
        - atol (float, default=0.01): absolute tolerance
    """
    output, tgt = None, None
    if num_additional_node > 0:
        output = out[:, :-num_additional_node]
        tgt = y[:, :-num_additional_node]
    else:
        output = out
        tgt = y
    num_close = (torch.isclose(output, tgt, rtol=rtol, atol=atol).sum(dim=1) == n).sum().item()
    return num_close / out.size(0)

def append_positional_encoding(x, pe):
    # Add positional encoding `pe` to input data `x`
    # Input `x` should have dimension [Batch, SeqLen, EmbedDim]
    # Input `pe` should have dimension [SeqLen, PEDim]
    # Output has dimension [Batch, Seqlen, EmbedDim + PEDim]
    pe = pe.unsqueeze(0)
    pe = torch.repeat_interleave(pe, x.size(0), dim=0)
    return torch.cat([x, pe], dim=-1)

def identity_pe(n):
    return torch.eye(n)

def get_pe(base_pe, x, num_additional_node):
    pos_enc = base_pe[:x.size(1)-num_additional_node]
    if num_additional_node > 0:
        pos_enc = torch.cat([pos_enc,base_pe[-num_additional_node:]], dim=0)
    return pos_enc

def binary_pe(n):
    bits = torch.ceil(torch.log2(torch.tensor(n))).int()
    pe = torch.zeros(n, bits)
    for i in range(n):
        for j in range(bits):
            pe[i, j] = (i >> j) & 1
    return 2*pe - 1

def sinusoidal_pe(n, dim):
    pe = torch.zeros(n, dim)
    position = torch.arange(0, n).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe
