# %% [markdown]
# # EDA for Higgs Boson Dataset

# %% [markdown]
# ## Import packages

# %%
from pyspark.sql import SparkSession
from pyspark import SparkContext
import os

# %% [markdown]
# ## Create SparkContext and SparkSession

# %%
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# %% [markdown]
# ## Read CSV, transform and load to Dataframe

# %%
RAW_DATA_FOLDER = "raw_data"
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
total_rows = df.count()
print(f"Total instances - {total_rows}")

# %%
col_name = "lepton pT"
limit = 4.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.0 to 4.0

# %%
col_name = "missing energy magnitude"
limit = 4.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.0 to 4.0

# %%
col_name = "jet 1 pT"
limit = 4.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.0 to 4.0

# %%
df.select("jet 1 b-tag").distinct().show()
# categorical variable, replace with 0, 1, 2 respectively

# %%
col_name = "jet 2 pT"
limit = 4.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.0 to 4.0

# %%
df.select("jet 2 b-tag").distinct().show()
# categorical variable, replace with 0, 1, 2 respectively

# %%
col_name = "jet 3 pT"
limit = 4.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.0 to 4.0

# %%
df.select("jet 3 b-tag").distinct().show()
# categorical variable, replace with 0, 1, 2 respectively

# %%
col_name = "jet 4 pT"
limit = 4.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.0 to 4.0

# %%
df.select("jet 4 b-tag").distinct().show()
# categorical variable, replace with 0, 1, 2 respectively

# %%
col_name = "m_jj"
limit = 7.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 7.0 to 7.0

# %%
col_name = "m_jjj"
limit = 4.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.0 to 4.0

# %%
col_name = "m_lv"
limit = 2.5
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 2.5 to 2.5

# %%
col_name = "m_jlv"
limit = 4.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.0 to 4.0

# %%
col_name = "m_bb"
limit = 4.5
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 4.5 to 4.5

# %%
col_name = "m_wbb"
limit = 3.5
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 3.5 to 3.5

# %%
col_name = "m_wwbb"
limit = 3.0
above = True
count = 0
if above:
    count += df.select(col_name).filter(df[col_name] > limit).count()
else:
    count += df.select(col_name).filter(df[col_name] < limit).count()
print(
    f"Count of instances with ({col_name} {'>' if above else '<'} {limit}) - {count}"
)
print(
    f"Percentage of instances with ({col_name} {'>' if above else '<'} {limit}) - {(count * 100.0/ total_rows):.4f}%"
)
# clip values above 3.0 to 3.0


