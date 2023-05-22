import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
from sklearn.metrics import RandomForestClassifier
import datetime

np.random.seed(456)

_, (train, valid, test), _ = dc.molnet.load_tox21()