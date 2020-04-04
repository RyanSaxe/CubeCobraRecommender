# CubeCobraRecommender

Recommender System for [CubeCobra](https://cubecobra.com/).

*More explicit documentation on how this currently works coming soon.*

## Generating The Adjacency Matrix

running `python create_mtx.py` will create a local version of the adjacency matrix as well as a lookup dictionary. This will be stored in the `outputs` folder. It is in `.gitignore`, so make sure to create a local version.

## Generating Recommendations

After generating the adjacency matrix, given any cube list, you can get the top N recommendations. To do this, run `python recommend.py cube_id N`. For example, if I wanted the top 50 recommendations for my [Combat Cube](https://cubecobra.com/cube/list/combat), I would run `python recommend.py combat 50`.

If you would like a recommendation on cards to cut, rather than cards to add, run `python cut_cards.py cube_id N`.

Lastly, if you would like recommendations from the machine learning algorithm rather than the adjacency matrix, run `python ml_recommend.py cube_id N`

## Git - LFS

In order to upload the data used in this project, it was zipped and tracked via [git-lfs](https://git-lfs.github.com/). You may need to install this in order to download the repo.

## To Do

1. Add code for training the ML model (currerntly `ml_files` contains a wrapped pretrained version).
2. Clean up repository to have a better structure. Comment everything. Separate scripts from source.

