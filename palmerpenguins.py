import pandas as pd
import seaborn as sns
import bokeh
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.metrics import confusion_matrix

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
