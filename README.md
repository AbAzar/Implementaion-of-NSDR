# Semi-Supervised Graph Learning: Near Strangers or Distant Relatives

This code implements the Semi-Supervised Graph Learning algorithm as described in the article "Semi-supervised Graph Learning: Near Strangers or Distant Relatives." The algorithm is applied to a given dataset for clustering purposes. Here's a brief overview of the code:

## Prerequisites

Ensure that you have the required libraries installed:

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

## Usage

1. Import necessary libraries:

```python
from math import exp
import numpy as np
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy.linalg as linalg
import random
import scipy.sparse as sparse
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.special import comb
```

2. Load the dataset. Uncomment and modify the following lines based on your dataset:

```python
# df = pd.read_csv('iris_csv.csv', names=names, header=1).values
# df = pd.read_csv('dermatology.data').values
df = pd.read_csv('balance-scale.data').values
```

3. Configure the target index and preprocessing:

```python
# target_index = -1  # For iris or dermatology datasets
target_index = 0   # For balance scale dataset
le = preprocessing.LabelEncoder()
le.fit(df[:, target_index])
df[:, target_index] = le.transform(df[:, target_index])
df = df.astype('float64')
```

4. Implement the NSDR_Ncut function for clustering:

```python
def NSDR_Ncut(df, N, K, sigma, n):
    # Implementation details...
    return cluster_assignments
```

5. Evaluate the algorithm by computing the Rand Index:

```python
N = len(le.classes_)
K = 5
sigma = 1
n = 10
res = 0

for i in range(10):
    np.random.shuffle(df)
    label = df[:, -1].astype('int64')
    pred = NSDR_Ncut(df, N, K, sigma, n * 2)
    res += rand_index_score(pred, label)
    print(rand_index_score(pred, label))

print('\t', res / 10)
```

## Notes:

- Adjust the dataset loading and preprocessing steps based on your specific dataset.
- Fine-tune the parameters (N, K, sigma, n) for optimal clustering results.
- The Rand Index is used for evaluating the clustering performance.

Feel free to modify the code to suit your specific needs or experiment with different datasets and parameters.