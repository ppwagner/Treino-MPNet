import torch


def make_query_and_content_mask(tensor, a, b, kind='MPLM'):
    '''
        Query Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]
        Content Mask:
        | <-   PLM   -> |    | <-      MPNet    -> |
                               x x x x x x x m m m
                               1 2 3 4 5 6 7 5 6 7
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 1 1 1 ]    [ 0 0 0 0 1 1 1 0 0 0 ]
        [ 0 0 0 0 0 1 1 ]    [ 0 0 0 0 0 1 1 1 0 0 ]
        [ 0 0 0 0 0 0 1 ]    [ 0 0 0 0 0 0 1 1 1 0 ]
        [ 0 0 0 0 0 0 0 ]    [ 0 0 0 0 0 0 0 1 1 1 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]
                             [ 0 0 0 0 1 1 1 0 0 0 ]

    '''

    def make_query_mask():
        mask = torch.triu(torch.ones(b, b), 0)
        mask = (torch.ones(b, a - b), 1 - mask) if kind is 'PLM' else (torch.ones(b, a - b), 1 - mask, mask)
        return torch.cat(mask, dim=-1).eq(0)

    def make_content_mask():
        mask = [torch.zeros(a - b, b), torch.tril(torch.ones(b, b), 0)]
        if kind is not 'PLM':
            mask.append(torch.zeros(b, b))
        mask = torch.cat(mask, dim=0)
        mask = (torch.ones(a, a - b), mask) if kind is 'PLM' else (torch.ones(a + b, a - b), mask, 1 - mask)
        return torch.cat(mask, dim=-1).eq(0)

    return make_query_mask().to(tensor.device), make_content_mask().to(tensor.device)


def make_unified_square_mask(tensor, a, b, kind='MPLM'):
    """
    Generates a single square attention mask for permuted tokens.
    1. Generates Query and Content masks.
    2. Concatenates Query mask below Content mask.
    3. Concatenates ones to the right to ensure the final mask is square.
    """
    q_mask, c_mask = make_query_and_content_mask(tensor, a, b, kind)

    # Concatenate Query mask below the Content mask
    # c_mask shape: (Height_c, Width)
    # q_mask shape: (Height_q, Width)
    # combined shape: (Height_c + Height_q, Width)
    combined_mask = torch.cat([c_mask, q_mask], dim=0)

    # As the mask should be square, concatenate ones for this
    h, w = combined_mask.shape
    diff = h - w

    # final_mask = combined_mask
    if diff > 0:
        # Create a block of ones (True/Masked) to pad the width
        # Assuming Boolean masks where True = Masked, False = Attend
        ones = torch.ones(h, diff, device=tensor.device, dtype=combined_mask.dtype)
        final_mask = torch.cat([combined_mask, ones], dim=1)
    else:
        final_mask = combined_mask

    return final_mask == False
