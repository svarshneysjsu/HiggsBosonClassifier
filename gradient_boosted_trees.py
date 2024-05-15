# %% [markdown]
# # Gradient Boosted Trees for Higgs Boson Detection

# %% [markdown]
# ## Import packages

# %%
from pyspark.ml.pipeline import PipelineModel, Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.sql.functions import col, expr, broadcast
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark import SparkContext
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# %% [markdown]
# ## Create SparkContext and SparkSession

# %%
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# %%
CLEAN_DATA_FOLDER = "clean_data"
MODELS_FOLDER = "models"
TRAIN = False
SCHEMA = [
    "signal",
    "lepton pT",
    "lepton eta",
    "lepton phi",
    "missing energy magnitude",
    "missing energy phi",
    "jet 1 pt",
    "jet 1 eta",
    "jet 1 phi",
    "jet 1 b-tag",
    "jet 2 pt",
    "jet 2 eta",
    "jet 2 phi",
    "jet 2 b-tag",
    "jet 3 pt",
    "jet 3 eta",
    "jet 3 phi",
    "jet 3 b-tag",
    "jet 4 pt",
    "jet 4 eta",
    "jet 4 phi",
    "jet 4 b-tag",
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
]

# %% [markdown]
# ## Read Train CSV, transform and load to Dataframe

# %%
rdd = sc.textFile(os.path.join(CLEAN_DATA_FOLDER, "train.csv"))
rdd = rdd.map(
    lambda row: [
        int(float(v)) if i == 0 else float(v) for i, v in enumerate(row.split(","))
    ]
)
train_df = rdd.toDF(schema=SCHEMA)

# %% [markdown]
# ## Build Gradient Boosted Trees Model
# 
# if TRAIN is True, this will train a Gradient Boosted Trees Pipeline Model using train csv
# 
# if TRAIN is False, this will load a trained Gradient Boosted Trees Pipeline Model from `models`

# %%
model = None
if TRAIN:
    vector_assembler = VectorAssembler(
        inputCols=[c for c in SCHEMA if c != "signal"], outputCol="features"
    )
    classifier = GBTClassifier(featuresCol="features", labelCol="signal", maxIter=20, maxDepth=10)
    pipeline = Pipeline(stages=[vector_assembler, classifier])
    model = pipeline.fit(train_df)
    model.save(os.path.join(MODELS_FOLDER, "gradientBoostedTrees"))
else:
    model = PipelineModel.load(os.path.join(MODELS_FOLDER, "gradientBoostedTrees"))

# %% [markdown]
# ## Feature Importance

# %%
importances = model.stages[-1].featureImportances.toArray()
feature_names = model.stages[0].getInputCols()
feature_importances = list(zip(feature_names, importances))
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
feature_names, importances = zip(*feature_importances)
_, ax = plt.subplots(1, 1, figsize=(8, 6))
_ = sns.barplot(x=importances, y=feature_names, ax=ax)
ax.set_title("Feature Importance")

# %% [markdown]
# ## Build Binary Classification Evaluators and Visualizers

# %%
evaluators = {
    "accuracy": MulticlassClassificationEvaluator(
        labelCol="signal", predictionCol="prediction", metricName="accuracy"
    ),
    "precision": MulticlassClassificationEvaluator(
        labelCol="signal", predictionCol="prediction", metricName="weightedPrecision"
    ),
    "recall": MulticlassClassificationEvaluator(
        labelCol="signal", predictionCol="prediction", metricName="weightedRecall"
    ),
    "f1": MulticlassClassificationEvaluator(
        labelCol="signal", predictionCol="prediction", metricName="f1"
    ),
    "areaUnderROC": BinaryClassificationEvaluator(
        labelCol="signal", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ),
    "areaUnderPR": BinaryClassificationEvaluator(
        labelCol="signal", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ),
}
metrics = []

# %%
def evaluate(predictions, split):
    accuracy = evaluators["accuracy"].evaluate(predictions)
    precision = evaluators["precision"].evaluate(predictions)
    recall = evaluators["recall"].evaluate(predictions)
    f1 = evaluators["f1"].evaluate(predictions)
    auc_roc = evaluators["areaUnderROC"].evaluate(predictions)
    auc_pr = evaluators["areaUnderPR"].evaluate(predictions)
    return {
        "Split": split,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AreaUnderROC": auc_roc,
        "AreaUnderPR": auc_pr,
    }

# %%
def plot_confusion_matrix(predictions, split, ax):
    TP = predictions.filter(
        (predictions.prediction == 1) & (predictions.signal == 1)
    ).count()
    TN = predictions.filter(
        (predictions.prediction == 0) & (predictions.signal == 0)
    ).count()
    FP = predictions.filter(
        (predictions.prediction == 1) & (predictions.signal == 0)
    ).count()
    FN = predictions.filter(
        (predictions.prediction == 0) & (predictions.signal == 1)
    ).count()
    confusion_matrix = [[TP, FP], [FN, TN]]
    _ = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix for {split}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

# %%
def plot_precision_recall_curve(predictions, split, ax):
    slim_preds = predictions.select(["signal", "probability", "prediction"]).withColumn(
        "probability_array", vector_to_array("probability")
    )
    # Create a DataFrame of threshold values
    thresholds = np.linspace(0, 1, num=100)
    threshold_df = spark.createDataFrame(
        [(float(t),) for t in thresholds], ["threshold"]
    )
    cross_joined = slim_preds.crossJoin(broadcast(threshold_df))
    # Define the conditions for TP, FP, TN, FN
    metrics_df = (
        cross_joined.withColumn(
            "Metric",
            expr(
                "CASE "
                + " WHEN probability_array[1] >= threshold AND signal = 1 THEN 'TP' "
                + " WHEN probability_array[1] >= threshold AND signal = 0 THEN 'FP' "
                + " WHEN probability_array[1] < threshold AND signal = 1 THEN 'FN' "
                + " ELSE 'TN' END"
            ),
        )
        .groupBy("threshold")
        .pivot("Metric")
        .count()
    )
    # Calculate precision and recall
    result_df = metrics_df.withColumn(
        "Precision", col("TP") / (col("TP") + col("FP"))
    ).withColumn("Recall", col("TP") / (col("TP") + col("FN")))
    # Collect results for plotting if necessary
    result_data = (
        result_df.select("threshold", "Precision", "Recall")
        .orderBy("threshold")
        .collect()
    )
    # Plotting
    precisions = [
        row["Precision"] for row in result_data if row["Precision"] is not None
    ]
    recalls = [row["Recall"] for row in result_data if row["Recall"] is not None]
    # Filter out None values and ensure that both precision and recall are present
    filtered_data = [
        (row["Precision"], row["Recall"])
        for row in result_data
        if row["Precision"] is not None and row["Recall"] is not None
    ]
    # x[1] is the recall
    sorted_data = sorted(filtered_data, key=lambda x: x[1])
    # Unpack the sorted data
    precisions, recalls = zip(*sorted_data)
    average_precision = 0.0
    for i in range(1, len(precisions)):
        average_precision += (recalls[i] - recalls[i - 1]) * precisions[i]
    # Plotting
    ax.plot(recalls, precisions, color="tab:blue", alpha=0.7)
    ax.fill_between(recalls, precisions, color="tab:blue", alpha=0.2)
    ax.set_title(f"Precision-Recall Curve for {split} (AP={average_precision:.4f})")
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])

# %%
def plot_roc_curve(predictions, split, ax):
    slim_preds = predictions.select(["signal", "probability", "prediction"]).withColumn(
        "probability_array", vector_to_array("probability")
    )
    # Create a DataFrame of threshold values
    thresholds = np.linspace(0, 1, num=100)
    threshold_df = spark.createDataFrame(
        [(float(t),) for t in thresholds], ["threshold"]
    )
    cross_joined = slim_preds.crossJoin(broadcast(threshold_df))
    # Define the conditions for TP, FP, TN, FN
    metrics_df = (
        cross_joined.withColumn(
            "Metric",
            expr(
                "CASE "
                + " WHEN probability_array[1] >= threshold AND signal = 1 THEN 'TP' "
                + " WHEN probability_array[1] >= threshold AND signal = 0 THEN 'FP' "
                + " WHEN probability_array[1] < threshold AND signal = 1 THEN 'FN' "
                + " ELSE 'TN' END"
            ),
        )
        .groupBy("threshold")
        .pivot("Metric")
        .count()
    )
    metrics_summed = metrics_df.groupBy("threshold").sum("TP", "FP", "TN", "FN")
    # Calculate TPR and FPR
    result_df = metrics_summed.withColumn(
        "FPR", col("sum(FP)") / (col("sum(FP)") + col("sum(TN)"))
    ).withColumn("TPR", col("sum(TP)") / (col("sum(TP)") + col("sum(FN)")))
    # Collect results to the driver
    result_data = (
        result_df.select("threshold", "FPR", "TPR").orderBy("threshold").collect()
    )
    # Filter out None values from the results
    filtered_data = [
        row for row in result_data if row["FPR"] is not None and row["TPR"] is not None
    ]
    # Sort by FPR
    sorted_data = sorted(filtered_data, key=lambda x: x["FPR"])
    fprs = [row["FPR"] for row in sorted_data]
    tprs = [row["TPR"] for row in sorted_data]
    # Calculate AUC using the trapezoidal rule
    roc_auc = np.trapz(tprs, fprs)
    # Plotting
    ax.plot(
        fprs, tprs, color="tab:orange", lw=2, label=f"ROC Curve (Area={roc_auc:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="tab:blue", lw=2, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Receiver Operating Characteristic for {split}")
    ax.legend(loc="lower right")
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])

# %% [markdown]
# ## Predictions and Metrics for Train split

# %%
train_predictions = model.transform(train_df)

# %%
metrics.append(evaluate(predictions=train_predictions, split="Train"))

# %%
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# plot_confusion_matrix(train_predictions, "Train", ax)

# %%
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# plot_roc_curve(train_predictions, "Train", ax)

# %%
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# plot_precision_recall_curve(train_predictions, "Train", ax)

# %% [markdown]
# ## Read Test CSV, transform and load to Dataframe

# %%
rdd = sc.textFile(os.path.join(CLEAN_DATA_FOLDER, "test.csv"))
rdd = rdd.map(
    lambda row: [
        int(float(v)) if i == 0 else float(v) for i, v in enumerate(row.split(","))
    ]
)
test_df = rdd.toDF(schema=SCHEMA)

# %% [markdown]
# ## Predictions and Metrics for Test split

# %%
test_predictions = model.transform(test_df)

# %%
metrics.append(evaluate(predictions=test_predictions, split="Test"))

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_confusion_matrix(test_predictions, "Test", ax)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_roc_curve(test_predictions, "Test", ax)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_precision_recall_curve(test_predictions, "Test", ax)

# %% [markdown]
# ## Print Metrics

# %%
spark.createDataFrame(metrics).show(truncate=False)


