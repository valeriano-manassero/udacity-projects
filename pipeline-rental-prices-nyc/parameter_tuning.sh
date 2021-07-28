#!/bin/sh

hydra_options="modeling.max_tfidf_features=10,15,30 "\
"modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 "\
"modeling.random_forest.max_depth=15,30,60 "\
"modeling.random_forest.n_estimators=100,200,400 "\
"-m"

mlflow run . -P steps=train_random_forest -P hydra_options="$hidra_options"


#
mlflow run . -P steps=train_random_forest -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 modeling.random_forest.max_depth=15,30,60 modeling.random_forest.n_estimators=100,200,400 -m"
