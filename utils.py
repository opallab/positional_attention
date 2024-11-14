import torch

def get_loss(criterion, out, y, num_additional_node, n, target):
    if target == 'path':
        return criterion(torch.triu(out[:,:n,:], 1), torch.triu(y[:,:n,:], 1))
    else:
        if num_additional_node > 0:
            return criterion(out[:, :-num_additional_node], y[:, :-num_additional_node])
        else:
            return criterion(out, y)

def get_accuracy(out, y, num_additional_node, n, target):
    if target == 'path':
        raise NotImplementedError
    elif target == 'median':
        output, tgt = None, None
        if num_additional_node > 0:
            output = out[:, :-num_additional_node:2]
            tgt = y[:, :-num_additional_node:2]
        else:
            output = out[:, ::2]
            tgt = y[:, ::2]
        return ((output == tgt).sum(dim=1) == n // 2).sum().item() / out.size(0)
    else:
        output, tgt = None, None
        if num_additional_node > 0:
            output = out[:, :-num_additional_node]
            tgt = y[:, :-num_additional_node]
        else:
            output = out
            tgt = y
        num_equal = ((output == tgt).sum(dim=1) == n).sum().item()
        return num_equal / out.size(0)

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
