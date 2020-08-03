#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests

API_URL = 'https://cubecobra.com/tool/api'


def dump_objects(key: str, obj_type: str, start_index: int = 0) -> None:
    page = start_index
    page_count = page + 1
    Path(f'data/{obj_type}').mkdir(parents=True, exist_ok=True)
    prev_max = ''
    while page < page_count:
        response = None
        try:
            response = requests.get(f'{API_URL}/download{obj_type}s/{page}/{key}',
                                    params={'prevMax': prev_max})
            requests_json = response.json()
            page_count = requests_json.get('pages', 1)
            prev_max = requests_json.get('prevMax', '')
            with open(f'data/{obj_type}/{obj_type}.{page}.json', 'w') as out_file:
                out_file.write(json.dumps(requests_json.get(f'{obj_type}s', [])))
            page += 1
            logging.info(f'Downloaded {page} out of {page_count} pages of {obj_type}s')
        except Exception as e:
            if response is not None:
                logging.error(f'status code: {response.status_code}, body: {response.text}\n error: {e}')
            else:
                logging.error(e)
        time.sleep(5)


def dump_cubes(key: str, start_index: int = 0) -> None:
    dump_objects(key, 'cube', start_index)


def dump_decks(key: str, start_index: int = 0) -> None:
    dump_objects(key, 'deck', start_index)


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
    parser = argparse.ArgumentParser()
    parser.add_argument('key',
                        help='The API key for downloading from CubeCobra.')
    parser.add_argument('--cubes', type=int, const=0, nargs='?',
                        help='Download cube data from CubeCobra can specify the start index.')
    parser.add_argument('--decks', type=int, const=0, nargs='?',
                        help='Download deck data from CubeCobra.')
    args = parser.parse_args()
    if args.cubes is not None:
        dump_cubes(args.key, args.cubes)
    if args.decks is not None:
        dump_decks(args.key, args.decks)
