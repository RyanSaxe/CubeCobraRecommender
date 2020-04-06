import json
import numpy as np
import unidecode
from tensorflow.keras.models import load_model
import sys
import urllib.request


args = sys.argv[1:]
cube_name = args[0]
non_json = True
root = "https://cubecobra.com"
if len(args) > 1:
    amount = int(args[1])
    if len(args) > 2:
        root = args[2]
        non_json = False
else:
    amount = 100

print('Getting Cube List . . . \n')

url = root + "/cube/api/cubelist/" + cube_name

fp = urllib.request.urlopen(url)
mybytes = fp.read()

mystr = mybytes.decode("utf8")
fp.close()

card_names = mystr.split("\n")

print ('Loading Card Name Lookup . . . \n')

int_to_card = json.load(open('ml_files/recommender_id_map.json','r'))
int_to_card = {int(k):v for k,v in int_to_card.items()}
card_to_int = {v:k for k,v in int_to_card.items()}

num_cards = len(int_to_card)

print ('Creating Cube Vector . . . \n')

cube_indices = []
for name in card_names:
    idx = card_to_int.get(unidecode.unidecode(name.lower()))
    #skip unknown cards (e.g. custom cards)
    if idx is not None:
        cube_indices.append(idx)

cube = np.zeros(num_cards)
cube[cube_indices] = 1

print('Loading Model . . . \n')

model = load_model('ml_files/recommender.h5')

print ('Generating Recommendations . . . \n')

cube = np.array(cube,dtype=float).reshape(1,num_cards)
results = model(cube)[0].numpy()

ranked = results.argsort()[::-1]

output = dict()

recommended = 0
for rec in ranked:
    if cube[0][rec] != 1:
        card = int_to_card[rec]
        if non_json:
            print(card)
        else:
            output[card] = results[rec]
        recommended += 1
        if recommended >= amount:
            break

if not non_json:
    print('<result>',output,'</result>')
