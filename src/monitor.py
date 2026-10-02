"""Descriptive feature drift and missingness checks against training data."""
import numpy as np
import pandas as pd
from .data import FEATURES


def drift_report(reference, current):
    rows = []
    for name in FEATURES:
        old, new = reference[name].dropna(), current[name].dropna()
        psi = None
        if len(old) and len(new):
            edges = np.unique(np.quantile(old, np.linspace(0, 1, 6)))
            if len(edges) > 1:
                edges[0], edges[-1] = -np.inf, np.inf
                a = np.histogram(old, edges)[0].astype(float) + .5
                b = np.histogram(new, edges)[0].astype(float) + .5
                a /= a.sum(); b /= b.sum()
                psi = float(np.sum((b-a)*np.log(b/a)))
        rows.append(dict(feature=name, population_stability_index=psi,
                         training_missing_fraction=float(reference[name].isna().mean()),
                         current_missing_fraction=float(current[name].isna().mean())))
    return pd.DataFrame(rows)
