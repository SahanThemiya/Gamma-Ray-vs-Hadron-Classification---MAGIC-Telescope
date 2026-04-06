import os
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

LABEL_MAP = {1: "gamma", 0: "hadron"}
PALETTE = {1: "steelblue", 0: "tomato"}

def _save(name):
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{name}.png')
    plt.close()

def class_distribution(df):
    count = df["class"].map(LABEL_MAP).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    count.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    _save("class_dist")

def feature_distributions(df):
    features = [c for c in df.columns if c != "class"]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, col in zip(axes.flat, features):
        for label, grp in df.groupby("class"):
            grp[col].plot(kind="kde", ax=ax, label=LABEL_MAP[label], color=PALETTE[label], linewidth=1.8)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    fig.suptitle("Feature Distributions by Class", fontsize=14, y=1.01)
    _save("feature_distributions")


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
        ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    _save("correlation")


def pairplot_sample(df, n=1000, seed=42):
    # Subsample for speed — full 19k rows is noisy anyway
    sample = df.sample(n, random_state=seed)
    sample["label"] = sample["class"].map(LABEL_MAP)
    key_features = ["fAlpha", "fLength", "fWidth", "fSize", "fDist", "label"]
    g = sns.pairplot(sample[key_features], hue="label",
                     palette={"gamma": "steelblue", "hadron": "tomato"},
                     plot_kws={"alpha": 0.4, "s": 10}, diag_kind="kde")
    g.figure.suptitle("Pairplot — Key Hillas Parameters", y=1.01)
    plt.savefig(f"{PLOT_DIR}/pairplot.png", dpi=120)
    plt.close()


