# abtest-mlops
This repository contains a design of a reliable hypothesis testing algorithm for the Brand Impact Optimiser(BIO) service and to determine whether a recent advertising campaign resulted in a significant lift in brand awareness.

Data exploration and implementation of the conventional p-value based algorithm and the sequential A/B testing algorithm in Python are included in the notebooks in this repository.

The data collected for this challenge has the following columns auction_id: the unique id of the online user who has been presented the BIO. In standard terminologies this is called an impression id. The user may see the BIO questionnaire but choose not to respond.

Machine learning Models

Logistic regression
Decision Tree
Xgboost
For each version of the data Data Splited the data into: *70% training *20% validation *10% test sets

The machine learning models were trained using 5-fold cross validation
