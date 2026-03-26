import pandas as pd
import numpy as np
from src.preprocessing import FEATURE_COLS

EPS = 1e-8

COLS = [
    'aspect_ratio', 'ellipse_area', 'eccentricity',
    'conc_ratio', 'core_brightness',
    'asym_dist_ratio', 'dist_length_ratio',
    'sin_alpha', 'cos_alpha', 'alpha_dist',
    'moment_ratio', 'moment_magnitude',
    'log_length', 'log_area'
]

def add_features(df):
    df = df.copy()
    df['aspect_ratio'] = df['fLength']/(df['fWidth']+EPS)
