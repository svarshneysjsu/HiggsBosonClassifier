# HiggsBosonClassifier
Classifier model that can accurately distinguish Higgs boson and background noise.

For implementing the Higgs boson classifier, we implemented the python notebooks described below.

eda1_plots.ipynb – In this notebook, basic statistics of the dataset are analyzed along with it pairwise scatter plot. Further, heatmap is generated to identify the correlation between the features and select the best features. 

 eda2_feature_analysis.ipynb – In this notebook, we have analyzed the distribution of all the features using density plots and tried to figure out whether the features have useful information to differentiate between the two classes 0 and 1. 

eda3_outlier_detection.ipynb – In this notebook, outliers for each feature are detected and analyzed based on their importance and then they are removed in the data cleaning notebook. 

data_cleaning.ipynb - In this notebook, the outlier in every feature is clipped based on their threshold value to make the data even for better classification. Later the feature analysis of all the columns is done to understand its distribution. 

train_test_split.ipynb - In this notebook, the dataset is split into train and test data frames with a split ration of 70:30 and split data frames are then saved in train.csv and test.csv. 

logistic_regression_baseline.ipynb - In this notebook, logistic regression baseline model is trained with raw dataset, and it's evaluated on metrices such as Accuracy, Area under PR, Area under ROC, F1 score, Precision and Recall for both train and test dataset. 

logistic_regression_PCA.ipynb - In this notebook, logistic regression model is trained after doing PCA, and it's evaluated on metrices such as Accuracy, Area under PR, Area under ROC, F1 score, Precision and Recall for both train and test dataset. 

decision_tree.ipynb - In this notebook, decision trees classifier model is trained with cleaned dataset, feature importance is measured, and it's evaluated on metrices such as Accuracy, Area under PR, Area under ROC, F1 score, Precision and Recall for both train and test dataset. 

decision_tree_pca.ipynb – In this notebook, decision tree is trained after implementing PCA and is evaluated with all the metrics. 

random_forest.ipynb - In this notebook, random forest classifier model is trained with cleaned dataset, feature importance is measured, and it's evaluated on metrices such as Accuracy, Area under PR, Area under ROC, F1 score, Precision and Recall for both train and test dataset. 

gradient_boosted_trees.ipynb - In this notebook, gradient boost trees classifier model is trained with cleaned dataset, feature importance is measured, and it's evaluated on metrices such as Accuracy, Area under PR, Area under ROC, F1 score, Precision and Recall for both train and test dataset. 

inference.py - In this notebook, The inferencing is done to detect the hoggs boson particle for new data.
