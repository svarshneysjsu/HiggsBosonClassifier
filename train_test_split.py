# %% [markdown]
# # Train-Test Split for Higgs Boson Dataset

# %% [markdown]
# ## Import packages

# %%
from pyspark.sql import SparkSession
from pyspark import SparkContext
import shutil
import os

# %% [markdown]
# ## Create SparkContext and SparkSession

# %%
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# %%
STAGING_DATA_FOLDER = "staging_data"
CLEAN_DATA_FOLDER = "clean_data"

# %% [markdown]
# ## Read CSV, transform and load to Dataframe

# %%
rdd = sc.textFile(os.path.join(STAGING_DATA_FOLDER, "clean.csv"))
rdd = rdd.map(
    lambda row: [
        int(float(v)) if i == 0 else float(v) for i, v in enumerate(row.split(","))
    ]
)
df = rdd.toDF(
    schema=[
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
)

# %% [markdown]
# ## Split the Dataframe into 70:30 for Train and Test respectively

# %%
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# %% [markdown]
# ## Save the Train Data

# %%
train_df.coalesce(1).write.format("csv").option("header", "false").mode(
    "overwrite"
).save(os.path.join(CLEAN_DATA_FOLDER, "train"))
files = os.listdir(os.path.join(CLEAN_DATA_FOLDER, "train"))
csv_file = next((file for file in files if file.endswith(".csv")), None)
os.rename(
    os.path.join(CLEAN_DATA_FOLDER, os.path.join("train", csv_file)),
    os.path.join(CLEAN_DATA_FOLDER, "train.csv"),
)
shutil.rmtree(os.path.join(CLEAN_DATA_FOLDER, "train"))

# %% [markdown]
# ## Save the Test Data

# %%
test_df.coalesce(1).write.format("csv").option("header", "false").mode(
    "overwrite"
).save(os.path.join(CLEAN_DATA_FOLDER, "test"))
files = os.listdir(os.path.join(CLEAN_DATA_FOLDER, "test"))
csv_file = next((file for file in files if file.endswith(".csv")), None)
os.rename(
    os.path.join(CLEAN_DATA_FOLDER, os.path.join("test", csv_file)),
    os.path.join(CLEAN_DATA_FOLDER, "test.csv"),
)
shutil.rmtree(os.path.join(CLEAN_DATA_FOLDER, "test"))


