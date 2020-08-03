import json
import os
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
        'invalid card',
    ]
    BAD_FUNCTIONS = [
        lambda x: x.get('isToken'),
    ]
    with open(card_file, 'rb') as cf:
        card_dict = json.load(cf)
    for cd in card_dict.values():
        for bf in BAD_FUNCTIONS:
            if bf(cd):
                BAD_NAMES.append(cd.get['name_lower'])
    return BAD_NAMES


def get_card_maps(map_file, exclude_file=None):
    exclusions = exclude(exclude_file)
    with open(map_file, 'rb') as mf:
        names = json.load(mf)
    name_lookup = dict()
    card_to_int = dict()
    int_to_card = dict()
    num_cards = 0
    for name, ids in names.items():
        if name in exclusions:
            continue
        card_to_int[name] = num_cards
        for idx in ids:
            name_lookup[idx] = name
        num_cards += 1
    int_to_card = {v: k for k, v in card_to_int.items()}
    return (
        num_cards,
        name_lookup,
        card_to_int,
        int_to_card
    )


def get_num_cubes(cube_folder, require_side=False):
    num_cubes = 0
    for f in os.listdir(cube_folder):
        full_path = os.path.join(cube_folder, f)
        with open(full_path, 'rb') as fp:
            contents = json.load(fp)
        if require_side:
            num_cubes += len([x for x in contents if len(x['side']) > 0])
        else:
            num_cubes += len(contents)
    return num_cubes


def build_cubes(cube_folder, num_cubes, num_cards,
                name_lookup, card_to_int):
    cubes = np.zeros((num_cubes, num_cards))
    counter = 0
    for f in os.listdir(cube_folder):
        full_path = os.path.join(cube_folder, f)
        with open(full_path, 'rb') as fp:
            contents = json.load(fp)
        for cube in contents:
            card_ids = []
            for card_name in cube:
                if card_name is not None:
                    card_id = card_to_int.get(card_name)
                    if card_id is not None:
                        card_ids.append(card_id)
            cubes[counter, card_ids] = 1
            counter += 1
    return cubes


def build_decks(cube_folder, num_cubes, num_cards,
                name_lookup, card_to_int, require_side=False):
    cubes = np.zeros((num_cubes, num_cards))
    counter = 0
    for f in os.listdir(cube_folder):
        full_path = os.path.join(cube_folder, f)
        with open(full_path, 'rb') as fp:
            contents = json.load(fp)
        for cube in contents:
            card_ids = []
            if require_side and len(cube['side']) == 0:
                continue
            for card_name in cube['main']:
                if card_name is not None:
                    card_id = card_to_int.get(card_name)
                    if card_id is not None:
                        card_ids.append(card_id)
            weight = 1
            if len(cube['side']) == 0:
                weight = 0.5
            cubes[counter, card_ids] = weight
            counter += 1
    return cubes


def create_adjacency_matrix(cubes, verbose=True, force_diag=None):
    num_cards = cubes.shape[1]
    adj_mtx = np.empty((num_cards, num_cards))
    for i in range(num_cards):
        if verbose:
            if i % 100 == 0:
                print(i+1, "/", num_cards)
        idxs = np.where(cubes[:, i] > 0)
        cubes_w_cards = cubes[idxs]
        step1 = cubes_w_cards.sum(0)
        if step1[i] != 0:
            step2 = step1/step1[i]
        else:
            step2 = step1
        adj_mtx[i] = step2
    if force_diag is not None:
        np.fill_diagonal(adj_mtx, force_diag)
    return adj_mtx
