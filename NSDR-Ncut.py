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

# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# df = pd.read_csv('iris_csv.csv',names =names, header=1).values

df = pd.read_csv('balance-scale.data').values

# df = pd.read_csv('dermatology.data').values
# df[df == '?'] = 0

# iris, dermatadology
# target_index = -1

# balance scale
target_index = 0

le = preprocessing.LabelEncoder()
le.fit(df[:, target_index])
df[:, target_index] = le.transform(df[:, target_index])
df = df.astype('float64')


def cal_sim_M(df, S, N, n):
    # print("Constricting M ...")
    M = [(i, j) for i in range(len(df)) for j in range(len(df)) if df[i, -1] == df[j, -1] if not i == j]
    random.shuffle(M)
    M = M[:n]
    # print("Calculating step1 ...")

    # step 1
    m1 = [(a[0], b[0]) for a in M for b in M if a[1] == b[1] and a[0] != b[0] and (a[0], b[0]) not in M and (b[0], a[0]) not in M]
    m2 = [(a[0], b[1]) for a in M for b in M if a[1] == b[0] and a[0] != b[1] and (a[0], b[1]) not in M and (b[1], a[0]) not in M]
    m3 = [(a[1], b[0]) for a in M for b in M if a[0] == b[1] and a[1] != b[0] and (a[1], b[0]) not in M and (b[0], a[1]) not in M]
    m4 = [(a[1], b[1]) for a in M for b in M if a[0] == b[0] and a[1] != b[1] and (a[1], b[1]) not in M and (b[1], a[1]) not in M]
    M += m1 + m2 + m3 + m4
    #
    # m = [(a[i], b[j]) for a in M
    #                     for b in M
    #                         for i in range(2)
    #                             for j in range(2)
    #                                 if a[(i+1)%2] == b[(j+1)%2]
    #                                     and a[i] != b[j]
    #                                         and (a[i], b[j]) not in M
    #                                             and (a[(i+1)%2], b[(j+1)%2]) not in M]
    #
    # M += m
    # for t1 in M:
    #     for t2 in M:
    #         if t1 == t2:
    #             continue
    #         a, b = -1, -1
    #         if t1[0] == t2[0]:
    #             a, b = t1[1], t2[1]
    #         if t1[1] == t2[0]:
    #             a, b = t1[0], t2[1]
    #         if t1[0] == t2[1]:
    #             a, b = t1[1], t2[0]
    #         if t1[1] == t2[1]:
    #             a, b = t1[0], t2[0]
    #         if a != -1:
    #             if (a, b) not in M or (b, a) not in M:
    #                 M.append((a, b))
    #         else:
    #             continue
    # print("Calculating step2 ...")
    # step 2
    for t in M:
        S[t[0], t[1]] = 1
        S[t[1], t[0]] = 1
    # print("Calculating step3 ...")
    # step 3
    for t in M:
        n1 = N[t[0]]
        n2 = N[t[1]]
        for i in n1:
            for j in n2:
                if i != j:
                    S[i, j] = max(S[i, j], S[i, t[0]] * S[t[0], t[1]] * S[j, t[1]])
                    S[j, i] = S[i, j]
    return M, S


def cal_dis_C(df, S, N, n):
    # print("Constricting C ...")
    C = [(i, j) for i in range(len(df)) for j in range(len(df)) if df[i, -1] != df[j, -1]]
    random.shuffle(C)
    C = C[:n]

    # print("Calculating step1 ...")
    for t in C:
        S[t[0], t[1]] = 0
        S[t[1], t[0]] = 0
    # print("Calculating step2 ...")
    for t in C:
        n1 = [i for i in N[t[0]] if i not in N[t[1]]]
        n2 = [i for i in N[t[1]] if i not in N[t[0]]]
        for i in n1:
            for j in n2:
                if i != j:
                    S[i, j] = min(S[i, j], 1 - S[i, t[0]] * S[j, t[1]])
                    S[j, i] = S[i, j]
    return C, S


# df: dataset
# N: number of clusters
# K: K nearest neighbor
# sigma
# n: number of pairwise constraints
def NSDR_Ncut(df, N, K, sigma, n):
    if target_index < 0:
        feature_x = df[:, :target_index]
    else:
        feature_x = df[:, (target_index + 1):]
    label_y = df[:, target_index]
    S = np.zeros((feature_x.shape[0], feature_x.shape[0]))
    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(feature_x, label_y)
    G = classifier.kneighbors_graph(feature_x)
    knn_dis, knn_index = classifier.kneighbors(feature_x, n_neighbors=K)
    shortest_dis = dijkstra(G, False)
    for i in range(len(knn_dis)):
        for j in range(len(knn_dis)):
            S[i, j] = exp(-((shortest_dis[i, j])**2)/2*sigma**2)
    M, S = cal_sim_M(df, S, knn_index, n)
    C, S = cal_dis_C(df, S, knn_index, n)
    a = np.dot(S, np.ones((S.shape[0], 1))).ravel()
    D = np.diag(a)
    W = np.dot(np.linalg.pinv(D), S)
    values, vectors = eigh(W, eigvals=(W.shape[0] - N, W.shape[0] - 1))
    # print(values)
    # print(vectors)

    kmeans = KMeans(n_clusters=N).fit(vectors)
    # print(W)

    cluster_assignments = kmeans.labels_
    return cluster_assignments


def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


N = len(le.classes_)
K = 5
sigma = 1
n = 10
res = 0
for i in range(10):
    np.random.shuffle(df)
    label = df[:, -1].astype('int64')
    pred = NSDR_Ncut(df, N, K, sigma, n*2)
    res += rand_index_score(pred, label)
    print(rand_index_score(pred, label))

print('\t', res / 10)
