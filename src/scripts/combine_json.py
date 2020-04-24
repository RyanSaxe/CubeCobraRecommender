#!/usr/bin/env python
import argparse
import glob
import json


def main(glob_pattern, out_file_name):
    results = []
    for file_name in glob.glob(glob_pattern):
        with open(file_name, 'r') as json_file:
            results += json.loads(json_file.read())
    with open(out_file_name, 'w') as json_file:
        json_file.write(json.dumps(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine several json array files into one.')
    parser.add_argument('glob_pattern',
                        help='Glob pattern to use to select the files. You will need to wrap this in quotes so the shell will not escape it.')
    parser.add_argument('out_file_name',
                        help='File name to put the results in.')
    args = parser.parse_args()
    main(**vars(args))
