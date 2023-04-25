import pandas as pd
import seaborn as sns
import bokeh

df = pd.read_csv(palmerpenguins.csv)
df.shape
df.columns

df.dtypes
df.isna().sum()
df.duplicated().sum()

df.describe()

corr = dataset.corr()
sns.heatmap(corr, 
            cmap="Spectral",
            fmt=".2f",
            annot=True)

sns.pairplot(dataset,hue="species")

sns.countplot(data=df,
    x="body_mass_g",
    hue="species"
    col="species",
    row="sex")

sns.lmplot(data=df,
    x="flipper_length_mm",
    y="body_mass_g",
    hue="species"
    col="species",
    row="sex")
