# %%
import glob
from pathlib import Path
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# %%
PATH = "../data/"

file_paths = glob.glob(os.path.join(PATH , "*.csv"))

# Load datasets
# File basename with dataframes
dfs = {}

for file_path in file_paths:
    dfs[Path(file_path).stem] = pd.read_csv(file_path)

sorted(list(dfs.keys()))

# Remove pandas warning
import warnings
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# %% [markdown]
# Information about the datasets can be found here: [CSV File Data Dictionary · synthetichealth/synthea Wiki](https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary)
#
# Notes: supplies dataset is empty

# %% [markdown]
# ### Finding similar patterns among patients who have three most common conditions
#
# Note: I don't have a background in health care, so there could be wrong
# statements in this notebook

# %%
conditions_df = dfs["conditions"]

# 3 most common conditions
conditions_agg = conditions_df.groupby(
    ["DESCRIPTION"], as_index=False).agg(
        count=pd.NamedAgg(column="DESCRIPTION", aggfunc="count"),).sort_values('count')[-3:]

fig = px.histogram(conditions_agg, y="DESCRIPTION", x="count")

fig.update_layout(
    title_text="Most Common Conditions",
    xaxis_title_text="Count", # yaxis label
)

fig.show()

# %% [markdown]
#
# **Sinusitis**, also known as rhinosinusitis, is inflammation of the mucous membranes that line the sinuses resulting in symptoms that may include thick nasal mucus, a plugged nose, and facial pain [Sinusitis - Wikipedia](https://en.wikipedia.org/wiki/Sinusitis)
#
# **Pharyngitis** is inflammation of the back of the throat, known as the pharynx [Pharyngitis - Wikipedia](https://en.wikipedia.org/wiki/Pharyngitis)
#
# **Acute bronchitis**, also known as a chest cold, is short-term bronchitis – inflammation of the bronchi (large and medium-sized airways) of the lungs [Acute bronchitis - Wikipedia](https://en.wikipedia.org/wiki/Acute_bronchitis)
#
# The common conditions seem to be related to an inflammation of a part of the respiratory system

# %% [markdown]
#
# Let's see how these conditions are treated using medications and procedures

# %%
# Get rows that are related to the common conditions
pd.set_option('display.max_colwidth', None)

medications_df = dfs['medications']

medications_df = medications_df[medications_df['REASONDESCRIPTION'].isin(list(conditions_agg['DESCRIPTION']))]

# 3 most common conditions
medications_agg = medications_df.groupby( ["REASONDESCRIPTION", "DESCRIPTION"], as_index=False).agg( 
    count=pd.NamedAgg(column="DESCRIPTION", aggfunc="count"),
).sort_values('count', ascending=False)

medications_agg.head()


# %%

procedures_df = dfs['procedures']
procedures_df = procedures_df[procedures_df['REASONDESCRIPTION'].isin(list(conditions_agg['DESCRIPTION']))]

# 3 most common conditions
procedures_agg = procedures_df.groupby( ["REASONDESCRIPTION", "DESCRIPTION"], as_index=False).agg( 
    count=pd.NamedAgg(column="DESCRIPTION", aggfunc="count"),
).sort_values('count', ascending=False)

procedures_agg.head()

# %% [markdown]
#
# Other common patterns among common conditions (duration of conditions, observations)

# %%

# The average duration of the conditions
def set_datetime_columns(df):
    df.loc[:, 'START'] = pd.to_datetime(df['START'])
    df.loc[:, 'STOP'] = pd.to_datetime(df['STOP'])
    return df

conditions_df = dfs['conditions']
conditions_df = set_datetime_columns(conditions_df.copy())

# Some conditions don't have an end time
conditions_df = conditions_df.dropna(subset=['STOP'])

conditions_df["Duration"] = conditions_df['STOP'] - conditions_df['START']

conditions_agg = conditions_df.groupby( ["DESCRIPTION"], as_index=False).agg(
    count=pd.NamedAgg(column="DESCRIPTION", aggfunc="count"),
    duration_avg=pd.NamedAgg(column="Duration", aggfunc="mean"),
).sort_values('count')[-3:]

conditions_agg["duration_avg"] = conditions_agg["duration_avg"]

conditions_agg.sort_values('count', ascending=False).head()
# %%

observations_df = dfs['observations']
observations_df.loc[observations_df['TYPE'] == 'numeric', 'VALUE'] = pd.to_numeric(observations_df[observations_df['TYPE'] == 'numeric']["VALUE"])
observations_df = observations_df[observations_df['TYPE'] == 'numeric']

conditions_df = dfs['conditions'][dfs['conditions']['DESCRIPTION'].isin(list(conditions_agg['DESCRIPTION']))]

conditions_observations_df = conditions_df.merge(
    observations_df, left_on = 'ENCOUNTER', right_on = 'ENCOUNTER',
    suffixes=('_condition', '_observation'))

conditions_observations_agg = conditions_observations_df.groupby( ["DESCRIPTION_condition", "DESCRIPTION_observation"], as_index=False).agg(
    observations_count=pd.NamedAgg(column="PATIENT_observation", aggfunc="count"),
    value_avg=pd.NamedAgg(column="VALUE", aggfunc=lambda x: x.mean()),
)

pd.set_option('display.max_rows', conditions_observations_agg.shape[0]+1)
conditions_observations_agg.sort_values('DESCRIPTION_observation')

# %%
fig = go.Figure()

for condition in list(conditions_agg['DESCRIPTION']):
    df = conditions_observations_agg[
        conditions_observations_agg['DESCRIPTION_condition'] == condition]
    fig.add_trace(go.Bar(
        name=condition,
        orientation='h',
        y=df["DESCRIPTION_observation"],
        x=df["value_avg"], 
        customdata=df['observations_count'],
        hovertemplate=" Value: %{x}" "<br>Count : %{customdata}"
        )
    )
fig.update_layout(
    title_text="Some observations of common conditions", # title of plot
    height= 1000,
    width=1200,
    xaxis_title_text="Average value", # xaxis label
    yaxis_title_text="Observations", # yaxis label
    legend_title="Conditions"
)


fig.show()

# %% [markdown]
# Machine leanring questions about the dataset
#
# 1- How can the data about patients be used to predict potential diseases they might have? (e.g. from observations and imaging_studies datasets, what're the probabilities for having a set of conditions)
#
# 2- How can the most effective treatment be picked for a patient with a certain set of conditions?
#
# 3- What's the probability for a condition to reoccur for a certain patient?
