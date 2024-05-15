from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.functions import col, when
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import sys
import os


MODEL_DATA_FOLDER = "models"
FINAL_MODEL = "gradientBoostedTrees"


def main(filepath):
    # read the input inference files
    df = spark.read.option("header", True).csv(filepath, header=True)
    # convert the datatypes
    for column in df.columns:
        df = df.withColumn(column, col(column).cast("float"))
    # load the model to detect Higgs Boson Particle
    model = PipelineModel.load(os.path.join(MODEL_DATA_FOLDER, FINAL_MODEL))
    # make predictions
    predictions = model.transform(df)
    # print the predictions
    predictions = predictions.select("prediction").withColumn(
        "prediction",
        when(predictions["prediction"] > 0.0, "Higgs Boson").otherwise(
            "Background Noise"
        ),
    )
    _ = predictions.show(truncate=False)


if __name__ == "__main__":
    # set the memory for large inference file
    conf = SparkConf()
    conf.set("spark.driver.memory", "12g")
    # create spark session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # create spark context
    sc = SparkContext.getOrCreate()
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File does not exist at path {filepath}")
        spark.stop()
        sys.exit(1)
    main(filepath)
