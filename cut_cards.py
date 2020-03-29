import urllib.request
import sys
from rec_funcs import simple_cuts
import json
import numpy as np
import unidecode

args = sys.argv[1:]
cube_name = args[0]
if len(args) > 1:
    amount = int(args[1])
else:
    amount = 100

print('Getting Cube List . . . \n')

url = "https://cubecobra.com/cube/api/cubelist/" + cube_name

fp = urllib.request.urlopen(url)
mybytes = fp.read()

mystr = mybytes.decode("utf8")
fp.close()

card_names = mystr.split("\n")

print ('Loading Adjacency Matrix . . . \n')

adj_mtx = np.load('output/full_adj_mtx.npy')

print ('Loading Card Name Lookup . . . \n')

int_to_card = json.load(open('output/int_to_card.json','r'))
int_to_card = {int(k):v for k,v in int_to_card.items()}
card_to_int = {v:k for k,v in int_to_card.items()}

print ('Creating Cube Vector . . . \n')

cube_indices = []
for name in card_names:
    idx = card_to_int.get(unidecode.unidecode(name.lower()))
    #skip unknown cards (e.g. custom cards)
    if idx is not None:
        cube_indices.append(idx)

cube = np.zeros(adj_mtx.shape[1])
cube[cube_indices] = 1

print ('Generating Recommendations . . . \n')

recs = simple_cuts(cube, adj_mtx, int_to_card)

for i in range(amount):
    rec = recs[i]
    print(str(i + 1) + ":", rec)
