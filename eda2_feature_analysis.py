# %% [markdown]
# # EDA for Higgs Boson Dataset

# %% [markdown]
# ## Import packages

# %%
from pyspark.sql import SparkSession
from pyspark import SparkContext
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %% [markdown]
# ## Create SparkContext and SparkSession

# %%
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# %%
RAW_DATA_FOLDER = "raw_data"
FIGURES_FOLDER = "figures"

# %% [markdown]
# ## Read CSV, transform and load to Dataframe

# %%
rdd = sc.textFile(os.path.join(RAW_DATA_FOLDER, "HIGGS.csv"))
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
# ## Feature Analysis

# %%
def plot_and_save_figure(data, colname, huename):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    _ = sns.kdeplot(data, x=colname, fill=True, ax=ax[0])
    _ = sns.kdeplot(data, x=colname, hue=huename, fill=True, ax=ax[1])
    fig.savefig(os.path.join(FIGURES_FOLDER, f"{colname}_before_cleaning.png"))

# %%
colname = "lepton pT"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "lepton eta"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "lepton phi"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "missing energy magnitude"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "missing energy phi"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 1 pt"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 1 eta"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 1 phi"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 1 b-tag"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 2 pt"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 2 eta"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 2 phi"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 2 b-tag"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 3 pt"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 3 eta"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 3 phi"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 3 b-tag"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 4 pt"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 4 eta"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 4 phi"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "jet 4 b-tag"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "m_jj"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "m_jjj"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "m_lv"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "m_jlv"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "m_bb"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "m_wbb"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")

# %%
colname = "m_wwbb"
data = df.select(["signal", colname]).sample(fraction=0.1, seed=42).toPandas()
plot_and_save_figure(data, colname, "signal")


