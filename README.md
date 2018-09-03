# Titanic-Passengers-Survival-Prediction

Dataset used for this project is from https://www.kaggle.com/c/titanic/data. This dataset is based on the one of the most infamous shipwrecks in history, the sinking of the Titanic. This is one the competition happening on Kaggle website for the analysis of what sorts of people were likely to survive.

The data has been split into two groups:

training set (train.csv)
test set (test.csv)
The training set should be used to build machine learning models. 

The test set should be used to see how well the model performs on unseen data. For each passenger in the test set, model which was trained is used to predict whether or not they survived the sinking of the Titanic.

Machine Learning solutions used:

Random Forest Classifier
SVM with different kernels (linear and rbf)
Gaussian Naïve Bayes

Data preprocessing:

Missing values replacement with median: Missing values in the data creates issue while vectorizing the data to retrieve feature matrix. Its necessary to remove these missing values. If we simply delete those examples from the data, we might incur in heavy loss of the data (especially for data with small data size) which are important to train the model.

Data discretization of continuous values: Since most of the features in our data have discretized values expect few ones. We should discretize these features with continuous values as these can mislead the network in understanding the data.

Finding the correlation between target feature and rest all features: Since our data has few features which are not relevant with our target feature, considering all the features will result in feature redundancy and feature explosion while training the model. Hence, its necessary to find the correlation between features to select the best features to create the feature matrix for training the model.

Machine learning techniques:

Splitting strategy for train and test set: The fundamental goal of machine learning is to generalize the model beyond the data used for training the model. Evaluation of model is estimated on the quality of its pattern generalization for data the model has not been trained on, that is unknown future instances of data for prediction. Since we need to check the accuracy of the model before it is applied to predict future instances that doesn’t have target values, we need to use some of the data for which we know the target values. Model evaluation with same data on which it is trained on is meaningless as it will simply learn those data and predict its target values. We need to generalize the model for which we need to keep some data in safe, unseen from the model while training. This data which is kept aside can be used to evaluate the efficiency of model for predicting future instances as we have target values for this data and these are not known to the model. Hence by using this splitting strategy, we can achieve generalized model which can do well on future data instances. In this project, 15% of the data was kept for test set which was kept unseen from the model till final evaluation and 85% of the data again got split in training and development to cross validate and tune the hyper-parameters.

Used cross validation to tune the hyper-parameters: Tuning the hyper parameters is very important step in machine learning. Hyperparameter optimization is the problem of choosing a set of optimal hyperparameters for a learning algorithm. These are not learnt during training of the model. These needs to passed to the algorithm. Deciding on the best parameter for given dataset and model is difficult. Hence, we do hyper-parameter tuning to achieve a better model.

Ensemble method: Ensemble modeling is a powerful way to improve the performance of the model. It combines several base models to produce one optimal predictive model. These are meta-algorithms as it combines several machine learning techniques into one predictive model in order to decrease variance, bias and improve predictions. This project uses Random Forest Classifier which is an ensemble algorithm as one of the classifier to get the optimal predictive model.
