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
# ## DataFrame Samples

# %%
df_sample = df.sample(fraction=0.000001, seed=42)
df_sample.toPandas().to_csv(
    os.path.join(FIGURES_FOLDER, "samples_before_cleaning.csv"), index=False
)
df_sample.show(truncate=False)

# %% [markdown]
# ## Basic Statistics of all columns

# %%
df_describe = df.describe()
df_describe.toPandas().to_csv(os.path.join(FIGURES_FOLDER, "stats_before_cleaning.csv"), index=False)
df_describe.show(truncate=False)

# %% [markdown]
# ## Pairwise Scatter Plot

# %%
fig = sns.pairplot(df.sample(fraction=0.00001, seed=42).toPandas(), hue="signal")
fig.savefig(os.path.join(FIGURES_FOLDER, "pairplot_before_cleaning.png"))

# %% [markdown]
# ## Heatmap for Pearson Correlation Coefficients

# %%
_, ax = plt.subplots(1, 1, figsize=(35, 32))
ax = sns.heatmap(
    df.select(df.columns).sample(fraction=0.00001, seed=42).toPandas().corr(),
    annot=True,
    vmax=1,
    vmin=-1,
    center=0.0,
    ax=ax,
)
ax.figure.savefig(os.path.join(FIGURES_FOLDER, "heatmap_before_cleaning.png"))

# %%
signal_counts = df.select(["signal"]).filter(df.signal == 1).count()
background_counts = df.select(["signal"]).filter(df.signal == 0).count()
total_counts = df.count()

# %%
print(f"Signal\t\t| count={signal_counts} proportion={signal_counts/total_counts:.4f}")
print(f"Background\t| count={background_counts} proportion={background_counts/total_counts:.4f}")


