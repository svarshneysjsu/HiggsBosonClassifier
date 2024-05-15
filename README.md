# HiggsBosonClassifier
Classifier model that can accurately distinguish Higgs boson and background noise.

## Expected folder structure

- root (/)
    - /clean_data
    - /staging_data
    - /raw_data
    - /figures
    - /models
    - baseline.ipynb
    - ...

## Download dataset

The dataset can be downloaded from https://archive.ics.uci.edu/static/public/280/higgs.zip

The HIGG.csv needs to be extracted and placed inside the `raw_data` folder.

## Order of execution

1. eda1_plots.ipynb
    - generates plots and figures which are saved in `figures`
2. eda2_feature_analysis.ipynb
    - generates plots and figures which are saved in `figures`
3. eda3_outlier_detection.ipynb
4. baseline.ipynb
    - To train the baseline model
5. data_cleaning.ipynb
    - This will clean `HIGGS.csv` from `raw data` and create `clean.csv` in `staging_data`
    - generates plots and figures which are saved in `figures`
6. train_test_split.ipynb
    - Consistently split `clean.csv` into 70:30 train-test split using seed=42, creates `train.csv` and `test.csv` in `clean_data`
7. logistic_regression.ipynb
8. logistic_regression_pca.ipynb
9. decision_tree.ipynb
10. decision_tree_pca.ipynb
11. random_forest.ipynb
12. gradient_boosted_trees.ipynb
13. inference.py
    - Command to execute inference.py - `spark-submit inference.py sample6.csv 2>/dev/null`

## Detailed description of code files below:

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

inference.py - In this script, The inferencing is done to detect the hoggs boson particle for new data.
