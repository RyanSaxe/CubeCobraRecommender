# CubeCobraRecommender

Recommender System for [CubeCobra](https://cubecobra.com/).

The `web` folder contains a Flask API that will return the Machine LearningCubeCobraRecommender recommendation in a json response. This is for integrating the machine learning algorithm on the website, which is currently in development as demonstrated in this [tweet](https://twitter.com/CubeCobra1/status/1247688818607771650). The version live on CubeCobra will exist in the `prod` branch.

## Machine Learning Algorithm

This recommender system is a Denoising AutoEncoder. There are multiple Dropout layers in the encoding step (including before the first hidden layer) to regularize the network. 

The only part of the algorithm that isn't commonly seen is an additional level of regularization via a Conditional Probability Graph. This can be done with an external datasource, or the same collection datasource. But the idea is as follows:

1. Take a dataset of items, and generate a matrix M such that M[i,j] is the probability that item j is in a collection given item i is in the collection.
2. Convert this matrix into rows of probabilities (the sum of every row should equal 1)

Then, use this matrix to regularize the AutoEncoder. The model is built of a couple parts:

1. Normal Encoder (E) 
2. Normal Decoder (D1)
3. Second Decoder (D2)

Let D be our dataset, and each data point d in D be a binary vector of length |M|. And let I be a the identity matrix of the same shape as M. Then, we optimize the following loss function:

Loss = BinaryCrossEntropy( D, D1(E(D)) ) + 0.001 * KL-Divergence( M, D2(E(I)) )

## Generating The Adjacency Matrix

running `python src/scripts/create_mtx.py` will create a local version of the adjacency matrix as well as a lookup dictionary. This will be stored in the `outputs` folder. It is in `.gitignore`, so make sure to create a local version.

## Generating Recommendations

After generating the adjacency matrix, given any cube list, you can get the top N recommendations. To do this, run `python src/scripts/recommend.py cube_id N`. For example, if I wanted the top 50 recommendations for my [Combat Cube](https://cubecobra.com/cube/list/combat), I would run `python src/scripts/recommend.py combat 50`.

If you would like a recommendation on cards to cut, rather than cards to add, run `python src/scripts/cut_cards.py cube_id N`.

Lastly, if you would like recommendations from the machine learning algorithm rather than the adjacency matrix, run `python src/scripts/ml_recommend.py cube_id N`

## Git - LFS

In order to upload the data used in this project, it was zipped and tracked via [git-lfs](https://git-lfs.github.com/). You may need to install this in order to download the repo.

## To Do

1. Add code for training the ML model (currerntly `ml_files` contains a wrapped pretrained version).
2. Clean up repository to have a better structure. Comment everything. Separate scripts from source.

