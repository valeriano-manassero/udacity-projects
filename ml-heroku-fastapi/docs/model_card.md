# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Valeriano Manassero created the model. It is Gradient Boosting Classifier using the default hyperparameters in scikit-learn.

## Intended Use

This model should be used to predict the salary of a person based off a some attributes about it's financials.

## Training Data

Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; training is done using 80% of this data.

## Evaluation Data


Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; evaluation is done using 20% of this data.

## Metrics

The model was evaluated using Accuracy score. The value is around 0.834.

## Ethical Considerations

Dataset contains data related race, gender and origin country. This will drive to a model that may potentially discriminate people; 
further investigation before using it should be done.

## Caveats and Recommendations

Given gender classes are binary (male/not male), which we include as male/female. Further work needed to evaluate across a
spectrum of genders.
