#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path

import requests


def dump_objects(key: str, type: str) -> None:
    page = 0
    page_count = 1
    Path(f'data/{type}').mkdir(parents=True, exist_ok=True)
    while page < page_count:
        try:
            response = requests.get(f'https://cubecobra.com/tool/api/download{type}s/{page}/{key}')
            requests_json = response.json()
            page_count = requests_json.get('pages', 1)
            with open(f'data/{type}/{type}.{page}.json', 'w') as out_file:
                out_file.write(json.dumps(requests_json.get(f'{type}s', [])))
            page += 1
        except Exception as e:
            logging.error(e)


def dump_cubes(key: str) -> None:
    dump_objects(key, 'cube')


def dump_decks(key: str) -> None:
    dump_objects(key, 'deck')


if __name__ == "__main__":
    key = sys.argv[1]
    dump_cubes(key)
    dump_decks(key)