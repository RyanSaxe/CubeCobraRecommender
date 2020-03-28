import json
import os
import sys
import numpy as np

def exclude(card_file=None):
    if card_file is None:
        return []
    BAD_NAMES = [
        'plains',
        'island',
        'swamp',
        'mountain',
        'forest',
        '1996 world champion',
    ]
    BAD_FUNCTIONS = [
        lambda x: True if x.get('isToken') else False,
    ]
    card_dict = json.load(open(card_file,'rb'))
    for cd in card_dict.values():
        for bf in BAD_FUNCTIONS:
            if bf(cd):
                BAD_NAMES.append(cd.get['name_lower'])
    return BAD_NAMES

def get_card_maps(map_file, exclude_file=None):
    exclusions = exclude(exclude_file)
    if exclude:
        exclusions = exclude
    names = json.load(open(map_file,'rb'))
    name_lookup = dict()
    card_to_int = dict()
    int_to_card = dict()
    num_cards = 0
    for name,ids in names.items():
        if name in exclusions:
            continue
        card_to_int[name] = num_cards
        for idx in ids:
            name_lookup[idx] = name
        num_cards += 1
    int_to_card = {v:k for k,v in card_to_int.items()}
    return (
        num_cards,
        name_lookup,
        card_to_int,
        int_to_card
    )

def get_num_cubes(cube_folder):
    num_cubes = 0
    for f in os.listdir(folder):
        full_path = os.path.join(folder,f)
        contents = json.load(open(full_path,'rb'))
        num_cubes += len(contents)
    return num_cubes

def build_cubes(cube_folder, num_cubes, num_cards):
    cubes = np.zeros((num_cubes,num_cards))
    counter = 0
    for f in os.listdir(folder):
        full_path = os.path.join(folder,f)
        contents = json.load(open(full_path,'rb'))
        for cube in contents:
            card_ids = []
            for card in cube['cards']:
                card_name = name_lookup.get(card['cardID'])
                if card_name is not None:
                    card_id = card_to_int.get(card_name)
                    if card_id is not None:
                        card_ids.append(card_id)
            cubes[counter,card_ids] = 1
            counter += 1
    return cubes

def create_adjacency_matrix(cubes, verbose=True, force_diag=0):
    num_cards = cubes.shape[1]
    adj_mtx = np.empty((num_cards,num_cards))
    for i in range(num_cards):
        if verbose:
            if i % 100 == 0:
                print(i+1,"/",num_cards)
        idxs = np.where(cubes[:,i] == 1)
        cubes_w_cards = cubes[idxs]
        step1 = cubes_w_cards.sum(0)
        if step1[i] != 0:
            step2 = step1/step1[i]
        else:
            step2 = step1
        adj_mtx[i] = step2
    if force_diag is not None:
        np.fill_diagonal(adj_mtx,force_diag)
    return adj_mtx

def get_all_recs(cubes, verbose=True):
    out = np.empty(cubes.shape)
    for i,cube in enumerate(cubes):
        if verbose:
            if i % 100 == 0:
                print(i+1,"/",cubes.shape[0])
        cube = cubes[i]
        out[i] = get_recs(cube)
    return out

def get_recs(cube, int_to_card=None):
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
        return [int_card[i] for i in rec_ids]





