import json
import numpy as np
import unidecode
from tensorflow.keras.models import load_model
import sys
import urllib.request

args = sys.argv[1:]
cube_name = args[0]
model_name = 'IKO'
non_json = True
root = "https://cubecobra.com"
if len(args) > 1:
    amount = int(args[1])
    if len(args) > 2:
        model_name = args[2]
        if len(args) > 3:
            root = args[3]
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

int_to_card = json.load(open('ml_files/iko_id_map.json','r'))
int_to_card = {int(k):v.lower() for k,v in int_to_card.items()}
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

model = load_model(f'ml_files/{model_name}')

# def encode(model,data):
#     return model.encoder.bottleneck(
#         model.encoder.encoded_3(
#             model.encoder.encoded_2(
#                 model.encoder.encoded_1(
#                     data
#                 )
#             )
#         )
#     )

# def decode(model,data):
#     return model.decoder.reconstruct(
#         model.decoder.decoded_3(
#             model.decoder.decoded_2(
#                 model.decoder.decoded_1(
#                     data
#                 )
#             )
#         )
#     )

def recommend(model,data):
    encoded = model.encoder(data,training=False)
    return model.decoder(encoded,training=False)

print ('Generating Recommendations . . . \n')

cube = np.array(cube,dtype=float).reshape(1,num_cards)
results = recommend(model,cube)[0].numpy()

ranked = results.argsort()[::-1]

output = {
    'additions':dict(),
    'cuts':dict(),
}

recommended = 0
for rec in ranked:
    if cube[0][rec] != 1:
        card = int_to_card[rec]
        if non_json:
            print(card)
        else:
            output['additions'][card] = results[rec].item()
        recommended += 1
        if recommended >= amount:
            break

for idx in cube_indices:
    card = int_to_card[idx]
    output['cuts'][card] = results[idx].item()

if non_json:
    cards = list(output['cuts'].keys())
    vals = list(output['cuts'].values())
    rank_cuts = np.array(vals).argsort()
    out = [cards[idx] for idx in rank_cuts[:amount]]
    print('\n')
    for i,item in enumerate(out): print(item,vals[rank_cuts[i]])
