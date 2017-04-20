# Learning Rank - IRDM 2017

Here is the source code for four learning to rank algorithms: RankNet, LambdaRank, LamdbaMART, and a logistic regression.

RankNet and LambdaRank are implemented in Tensorflow with the [models here](src/models.py) and the [training code here](src/ranknet.py). Examples of how to run the training code is [here](src/train.sh).

LambdaMART was run with Lemur, the settings are [here](src/lambda_mart_grid_search_1.sh), [here](src/lambda_mart_grid_search_2.sh), [here](src/lambda_mart_grid_search_3.sh), and [here](src/lambda_mart_grid_search_4.sh)

The logistic regression is [here](src/Logistic_Regressor.py).

# Authors

Joe Brown

Andrew Clarke

Anthony Hartshorn

Aseem Sharma
