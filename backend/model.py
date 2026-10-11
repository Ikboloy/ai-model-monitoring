"""A reproducible Iris model; monitoring concepts generalize to larger models."""
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
VERSION = 'iris-rf-v1'

@dataclass
class ModelBundle:
    classifier: object
    reference: np.ndarray
    accuracy: float
    classes: list


def train_model():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
    model.fit(x_train, y_train)
    return ModelBundle(model, x_train, float(accuracy_score(y_test, model.predict(x_test))), list(iris.target_names))


def population_stability_index(reference, current):
    """Quantile bins including both tails; smoothed histograms for finite PSI."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 6)))
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 2:
        return 0.0
    ref_counts = np.histogram(reference, bins=edges)[0] + 0.5
    cur_counts = np.histogram(current, bins=edges)[0] + 0.5
    ref_p = ref_counts / ref_counts.sum()
    cur_p = cur_counts / cur_counts.sum()
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
