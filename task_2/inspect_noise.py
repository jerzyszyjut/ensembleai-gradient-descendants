import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

first = []
second = []
third = []
fourth = []
fifth = []

iterations = []

PATH = "./task_2/labels"
for _, _, files in os.walk(PATH):
    for file in sorted(files):
        if file[0] == '.': continue
        print(file)
        arr = np.loadtxt(os.path.join(PATH, file), delimiter=",", dtype=np.float64)
        iterations.append(arr)

index = []
# Standard deviations of each output (not substracting)
# for i, iteration in enumerate(iterations):
#     index.append(i)
#     stds = np.std(iteration, axis=1)
#     first.append(stds[0])
#     second.append(stds[1])
#     third.append(stds[2])
#     fourth.append(stds[3])
#     fifth.append(stds[4])

# # Standard deviations (substracting)

subtract = iterations[0]
for i, iteration in enumerate(iterations):
    if i == 0: continue
    index.append(i)
    iteration = iteration - subtract
    stds = np.std(iteration, axis=1)
    first.append(stds[0])
    second.append(stds[1])
    third.append(stds[2])
    fourth.append(stds[3])
    fifth.append(stds[4])


plt.scatter(index, sorted(first))
plt.scatter(index, sorted(second))
plt.scatter(index, sorted(third))
plt.scatter(index, sorted(fourth))
plt.scatter(index, sorted(fifth))

first_sort = sorted(first)
trend_f = np.polyfit(index,first_sort,2)
trendpoly_f = np.poly1d(trend_f) 
plt.plot(index,trendpoly_f(index))

np.save("./linear_model.npy", trend_f)