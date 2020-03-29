import numpy as np

def simple_recs(cube, adj_mtx, int_to_card=None):
    cube_contains = np.where(cube == 1)[0]
    cube_missing = np.where(cube == 0)[0]
    sub_adj_mtx = adj_mtx[cube_contains][:,cube_missing]
    rec_ids = [
        cube_missing[i] for i  in
        sub_adj_mtx.sum(0).argsort()[::-1]
    ]
    if int_to_card is None:
        return rec_ids
    else:
        return [int_to_card[i] for i in rec_ids]

def simple_cuts(cube, adj_mtx, int_to_card=None):
    np.fill_diagonal(adj_mtx,0)
    cube_contains = np.where(cube == 1)[0]
    sub_adj_mtx = adj_mtx[cube_contains][:,cube_contains]
    rec_ids = [
        cube_contains[i] for i  in
        sub_adj_mtx.sum(0).argsort()
    ]
    if int_to_card is None:
        return rec_ids
    else:
        return [int_to_card[i] for i in rec_ids]

@DeprecationWarning
def dist_recs(cube, adj_mtx, dist_f, int_to_card, verbose=True):
    """
    very inefficient recommendation algorithm via computing
        the distance from every card in the cube to every other
        card. When testing this gave me the same output as 
        `simple_recs`, but was literally 100x slower. Do not
        Use this function. 
    """
    cube_contains = np.where(cube == 1)[0]
    cube_missing = np.where(cube == 0)[0]
    sub_adj_mtx = adj_mtx[cube_contains][:,cube_missing]
    out_mtx = np.empty(sub_adj_mtx.shape)
    n_cols = sub_adj_mtx.shape[1]
    mtx_for_dist = adj_mtx[cube_missing][:,cube_missing]
    for r_idx,row in enumerate(sub_adj_mtx):
        if verbose:
            if r_idx % 100 == 0:
                print(r_idx,"/",sub_adj_mtx.shape[0])
        new_row = np.empty(n_cols)
        for c_idx in range(n_cols):
            col = mtx_for_dist[:,c_idx]
            dist = dist_f(row,col)
            new_row[c_idx] = dist
        out_mtx[r_idx] = new_row
    rec_ids = [
        cube_missing[i] for i  in
        sub_adj_mtx.sum(0).argsort()
    ]
    return [int_to_card[i] for i in rec_ids]