# Higgs Boson Detection using PySpark

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