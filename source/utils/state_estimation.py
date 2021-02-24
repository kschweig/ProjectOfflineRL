import torch
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.decomposition import PCA


def estimate_randenc(states: torch.Tensor, reward: torch.Tensor, params, k=5, mesh=100, seed=42):

    # initialize seed for reproducability
    torch.manual_seed(seed)

    # random encoder
    encoder = nn.Sequential(
        nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=7 * 7 * 64, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=2)
    ).to(params.device)

    states = states.to(params.device)

    with torch.no_grad():
        states = encoder.forward(states).cpu().numpy()



    reward = reward.cpu().numpy()

    # knn distance measure
    proxies = states
    distances = []
    for i in range(len(proxies)):
        distance = (states - proxies[i].reshape(1,2))**2
        distance = np.sum(distance, axis=1)**0.5
        distance = distance[np.nonzero(distance)[0]]
        distance = distance[np.argpartition(distance, k)[:k]]
        distances.append(np.mean(distance).item())

    print(f"{params.experiment} dataset has mean distance to its {k} nearest neighbours of "
          f"{round(np.mean(distances), 6)}")

    # density measure
    occupied = np.zeros((mesh, mesh))
    step = 1.5 / mesh
    offset = 0.5

    for state in states:
        i = max(min(int((state[0] - offset) / step), int(mesh-1)), 0)
        j = max(min(int((state[1] - offset) / step), int(mesh-1)), 0)
        occupied[i, j] = 1

    print(f"{params.experiment} dataset has a density (Random Encoder) of {round(np.sum(occupied) / mesh**2, 4)}")

    print(f"Mean: {np.mean(states, axis=0)}, Std: {np.std(states, axis=0)}")

    plot(states, reward, "Random-Encoder", params.experiment, limit=True)


def estimate_sklearn(states: torch.Tensor, reward: torch.Tensor, params, mesh=100, seed=42):
    states = states.cpu().numpy().reshape(len(states), -1)
    reward = reward.cpu().numpy()

    pca = PCA(n_components=2, random_state=seed)
    x_pca = pca.fit_transform(states)
    plot(x_pca, reward, "PCA", params.experiment, limit=False)

    # density measure
    occupied = np.zeros((mesh, mesh))
    step_i = x_pca[:, 0].max() / mesh
    offset_i = x_pca[:, 0].min()
    step_j = x_pca[:, 1].max() / mesh
    offset_j = x_pca[:, 1].min()

    for state in x_pca:
        i = max(min(int((state[0] - offset_i) / step_i), int(mesh - 1)), 0)
        j = max(min(int((state[1] - offset_j) / step_j), int(mesh - 1)), 0)
        occupied[i, j] = 1

    print(f"{params.experiment} dataset has a density (PCA) of {round(np.sum(occupied) / mesh ** 2, 4)}")


    # use PCA to downproject first
    pca = PCA(n_components=50, random_state=seed)
    # use low number of neighbours to preserve local structure
    umap = UMAP(n_components=2, n_neighbors=20, random_state=seed)

    #x_umap = pca.fit_transform(states)
    x_umap = umap.fit_transform(states)
    plot(x_umap, reward, "UMAP", params.experiment, limit=False)

    # density measure
    occupied = np.zeros((mesh, mesh))
    step_i = (x_umap[:,0].max() - x_umap[:,0].min()) / mesh
    offset_i = x_umap[:,0].min()
    step_j = (x_umap[:, 1].max() - x_umap[:, 1].min()) / mesh
    offset_j = x_umap[:, 1].min()
    for state in x_umap:
        i = max(min(int((state[0] - offset_i) / step_i), int(mesh - 1)), 0)
        j = max(min(int((state[1] - offset_j) / step_j), int(mesh - 1)), 0)
        occupied[i, j] = 1

    print(f"{params.experiment} dataset has a density (UMAP) of {round(np.sum(occupied) / mesh ** 2, 4)}")


def plot(dim2_states: np.ndarray, y: np.ndarray, method: str, config: str, limit=False):

    # plot no more than 2000 points
    rng = np.random.default_rng(42)
    indices = rng.choice(np.arange(len(y)), 5000)
    dim2_states = dim2_states[indices]
    y = y[indices]

    color=[]
    for y_ in y:
        if y_ == 0:
            color.append("C2")
        else:
            color.append("C1")

    plt.figure()
    plt.title("Embedded Projection using "+method)
    plt.scatter(dim2_states[:,0], dim2_states[:,1], c=color, s=5, linewidths=0)
    if limit:
        plt.xlim(left=0.5, right=2)
        plt.ylim(bottom=0.5, top=2)
    plt.savefig(os.path.join("results", config, "projection_"+method+".pdf"))
    plt.show()