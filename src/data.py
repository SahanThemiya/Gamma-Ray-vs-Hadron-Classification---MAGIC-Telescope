import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLUMNS = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"
]

# Hillas parameters — image shape descriptors from Cherenkov shower ellipse
FEATURE_DESCRIPTIONS = {
    "fLength" : "Major axis of ellipse [mm]",
    "fWidth"  : "Minor axis of ellipse [mm]",
    "fSize"   : "log10 of total photon content",
    "fConc"   : "Ratio of two highest pixels to fSize",
    "fConc1"  : "Ratio of highest pixel to fSize",
    "fAsym"   : "Distance from highest pixel to center [mm]",
    "fM3Long" : "3rd root of 3rd moment along major axis [mm]",
    "fM3Trans": "3rd root of 3rd moment along minor axis [mm]",
    "fAlpha"  : "Angle of major axis with vector to origin [deg]",
    "fDist"   : "Distance from origin to center of ellipse [mm]",
}


def load(path="magic04.data"):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["class"] = (df["class"].str.strip() == "g").astype(int)  # 1=gamma, 0=hadron
    return df


def split(df, test_size=0.2, seed=42):
    X, y = df.drop("class", axis=1), df["class"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_te_s, y_tr, y_te, X.columns.tolist(), scaler
