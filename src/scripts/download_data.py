#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests


def dump_objects(key: str, type: str) -> None:
    page = 0
    page_count = 1
    Path(f'data/{type}').mkdir(parents=True, exist_ok=True)
    while page < page_count:
        try:
            API_URL = 'https://cubecobra.com/tool/api'
            response = requests.get(f'{API_URL}/download{type}s/{page}/{key}')
            requests_json = response.json()
            page_count = requests_json.get('pages', 1)
            with open(f'data/{type}/{type}.{page}.json', 'w') as out_file:
                out_file.write(json.dumps(requests_json.get(f'{type}s', [])))
            page += 1
            logging.info(f'Downloaded {page} out of {page_count} {type}s')
            time.sleep(0.25)
        except Exception as e:
            logging.error(e)


def dump_cubes(key: str) -> None:
    dump_objects(key, 'cube')


def dump_decks(key: str) -> None:
    dump_objects(key, 'deck')


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
    parser = argparse.ArgumentParser()
    parser.add_argument('key',
                        help='The API key for downloading from CubeCobra.')
    parser.add_argument('--cubes', action='store_true',
                        help='Download cube data from CubeCobra.')
    parser.add_argument('--decks', action='store_true',
                        help='Download deck data from CubeCobra.')
    args = parser.parse_args()
    if args.cubes:
        dump_cubes(args.key)
    if args.decks:
        dump_decks(args.key)
