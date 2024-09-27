import torch

def get_loss(criterion, out, y, num_additional_node, n, target):
    if target == 'path':
        return criterion(torch.triu(out[:,:n,:], 1), torch.triu(y[:,:n,:], 1))
    else:
        if num_additional_node > 0:
            return criterion(out[:, :-num_additional_node], y[:, :-num_additional_node])
        else:
            return criterion(out, y)

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
