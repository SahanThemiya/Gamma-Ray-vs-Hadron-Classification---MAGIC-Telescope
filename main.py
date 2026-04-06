import matplotlib
matplotlib.use('Agg')

import os
from src.data  import load, split
from src.eda   import class_distribution, feature_distributions, correlation_heatmap, pairplot_sample
from src.models import train_evaluate

os.makedirs("plots", exist_ok=True)

df = load("magic04.data")

print(f"Dataset shape : {df.shape}")
print(f"Class balance : {df['class'].value_counts().to_dict()}  (1=gamma, 0=hadron)\n")

class_distribution(df)
feature_distributions(df)
correlation_heatmap(df)
pairplot_sample(df)

X_train, X_test, y_train, y_test, feature_names, _ = split(df)

results, summary = train_evaluate(X_train, X_test, y_train, y_test, feature_names)