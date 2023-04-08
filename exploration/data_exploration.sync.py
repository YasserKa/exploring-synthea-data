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

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# %% [markdown]
# Information about the datasets can be found here: [CSV File Data Dictionary · synthetichealth/synthea Wiki](https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary)
#
# Notes: 
# - Supplies dataset is empty

# %% [markdown]
# ### Single patient trajectory
#
# The following presents the observations trajectory via a scatter plot
# %%
observations_df = dfs['observations']

# Trajectory for patient's observations
# Text data are transformed to numeric to present them in the same plot as the
# numeric data
observations_text = observations_df.loc[observations_df["TYPE"] == "text", "DESCRIPTION"].unique()

for observation in observations_text:
    cat_data = observations_df.loc[observations_df['DESCRIPTION'] == observation,'VALUE'].astype('category').cat.codes
    observations_df.loc[observations_df['DESCRIPTION'] == observation, 'VALUE'] = cat_data

observations_df['VALUE'] = observations_df['VALUE'].astype(float)

observations_all = observations_df["DESCRIPTION"].unique()
# Some values have different ranges. Add normalized_VALUE column and make a
# separate plot
for observation in observations_all:
    data = observations_df[observations_df["DESCRIPTION"] == observation]['VALUE']
    normalized_data = (data - data.min()) / (data.max() - data.min())
    observations_df.loc[observations_df["DESCRIPTION"] == observation, "normalized_VALUE"] = normalized_data

# Pick a random patient
patient_df = dfs['patients'].sample(n=1)
id = patient_df['Id'].iloc[0]
id = 'e061409e-4b85-4ec1-b1f7-02677d51f763'
observations_df_user = observations_df[observations_df['PATIENT'] == id].copy()

# Add encounter's description
observations_df_user = observations_df_user.merge(
    dfs["encounters"], left_on = "ENCOUNTER", right_on = "Id",
    suffixes=("_observation", "_encounter"))

fig = px.line(observations_df_user, x="DATE", y="VALUE",
              color="DESCRIPTION_observation", 
              custom_data=["DESCRIPTION_observation", "VALUE", "DESCRIPTION_encounter"], markers=True)


fig.update_layout(
    title_text="Observations trajectory",
    xaxis_title_text="Date", 
    yaxis_title_text="Value",
)

fig.update_traces(
    hovertemplate="<br>".join([
        "Observation: %{customdata[0]}",
        "Date: %{x}",
        "Value: %{y}",
        "Encounter description: %{customdata[2]}",
    ]),
    )
 
# Only enalbe a few traces by defaults to not overwhelm the plot
ENABLED_TRACES = ["Body Height", "Heart rate", "Respiratory rate"]
fig.for_each_trace(
    lambda trace: trace.update(visible="legendonly") if trace.name not in ENABLED_TRACES else (),
)
fig.show()

# %%
# Plot the figure with the normalized values
fig = px.line(observations_df_user, x="DATE", y="normalized_VALUE",
              color="DESCRIPTION_observation", 
              custom_data=["DESCRIPTION_observation", "VALUE", "DESCRIPTION_encounter"], markers=True)

fig.update_layout(
    height= 500,
    title_text="Normalized observations trajectory",
    xaxis_title_text="Date", 
    yaxis_title_text="Normalized Value",
)

fig.update_traces(
    hovertemplate="<br>".join([
        "Observation: %{customdata[0]}",
        "Date: %{x}",
        "Value: %{customdata[1]}",
        "Normalized Value: %{y}",
        "Encounter description: %{customdata[2]}",
    ]),
    )
 
fig.for_each_trace(
    lambda trace: trace.update(visible="legendonly") if trace.name not in ENABLED_TRACES else (),
)
fig.show()

# %%
def set_datetime_columns(df):
    df.loc[:, 'START'] = pd.to_datetime(df['START'], utc=True)
    df.loc[:, 'STOP_HOVER'] = df['STOP'].fillna("ONGOING")
    df.loc[df['STOP_HOVER'] == "ONGOING", 'STOP'] = pd.Timestamp.now()
    df.loc[:, 'STOP'] = pd.to_datetime(df['STOP'], utc=True)

    if str(df["START"].dt.tz) == "UTC":
        df['START'] = df['START'].dt.tz_convert(None)
    if str(df["STOP"].dt.tz) == "UTC":
        df['STOP'] = df['STOP'].dt.tz_convert(None)
    return df


# %% [markdown]
# Timeline showing the procedures dates and duration of conditions and
# medications for a patient
# %%
conditions_df = dfs["conditions"]

conditions_user_df = conditions_df.loc[conditions_df['PATIENT'] == id, ["START","STOP","CODE","DESCRIPTION"]]
conditions_user_df["type"] = "condition"

medications_df = dfs["medications"]
medications_user_df = medications_df.loc[medications_df['PATIENT'] == id, ["START","STOP","CODE","DESCRIPTION"]]
medications_user_df["type"] = "medication"

user_df = pd.concat([conditions_user_df, medications_user_df])
user_df = set_datetime_columns(user_df)

fig = px.timeline(
    user_df, x_start="START", x_end="STOP", y="DESCRIPTION", 
    hover_name = 'DESCRIPTION',
    custom_data = ['START', 'STOP_HOVER'],
    color = "type",
    opacity = 0.75,
    hover_data=['DESCRIPTION'],  height=400, width=1000, 
)
fig.update_layout(
    title_text="Patient Trajectory",
    plot_bgcolor = 'rgba(0,0,0,0)',
    width=1200,
)

fig.update_traces(
    hovertemplate="<br>".join([
        "START: %{customdata[0]}",
        "STOP: %{customdata[1]}",
    ])
    )

procedures_df = dfs["procedures"]
procedures_user_df = procedures_df.loc[procedures_df['PATIENT'] == id, ["DATE","CODE","DESCRIPTION"]]
procedures_user_df.loc[:, 'DATE'] = pd.to_datetime(procedures_user_df['DATE'], utc=True)
trace = go.Scatter(x=procedures_user_df['DATE'], y=len(procedures_user_df)*["Procedures"], mode='markers', customdata = procedures_user_df["DESCRIPTION"], hovertemplate="PROCEDURE: %{customdata}", showlegend=False)

fig.add_traces([trace])

fig.show()
# %% [markdown]
# ### Finding similar patterns among patients who have three most common conditions
#
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
conditions_df = dfs['conditions']
conditions_df = set_datetime_columns(conditions_df.copy())

# Some conditions don't have an end time
conditions_df = conditions_df.dropna(subset=['STOP'])

# The duration of the condition
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
