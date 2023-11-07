import skimage as ski
import matplotlib.pyplot as plt

image = ski.io.imread("logo_wide.png")


image = ski.color.rgb2gray(ski.color.rgba2rgb(image))


harris_image = ski.feature.corner_harris(image)
equalized_harris = ski.exposure.equalize_hist(harris_image)

import numpy as np

corner_peaks = ski.feature.corner_peaks(harris_image, threshold_rel=0.0000001)

# plt.imshow(image, cmap="gray")
corners = np.array(corner_peaks).T
# print(corners)
# plt.scatter(corners[1], corners[0])
# plt.figure()

fig,ax = plt.subplots(1,1)
ax.imshow(image, cmap="gray")

labels = {"m" : [], "i": [], "r":[], "a":[], ".": []}

letters = list(labels.keys())

plt.ion()

for point in corners.T:
    ax.clear()
    ax.imshow(image, cmap="gray")
    for l in letters:
        for i, p in enumerate(labels[l]):
            ax.scatter(p[1], p[0])
            ax.annotate(str(i), (p[1], p[0]), size=10, bbox=dict(boxstyle="square,pad=0.3",
                      fc="lightblue", ec="steelblue", lw=2))

    ax.scatter(point[1], point[0])
    ax.annotate("???", (point[1], point[0]), size=15, bbox=dict(boxstyle="square,pad=0.3",
                      fc="lightblue", ec="steelblue", lw=2))
    plt.pause(0.01)
    fig.show()
    l = input("Lettre ?")
    while l != "":
        if l in letters:
            n = input("numéro ?")
            labels[l].insert(int(n), point)
            break
        else:
            l = input("LETTRE ?")

import pickle

with open("mira_coords.pkl", "wb") as f:
    pickle.dump(labels, f)
    f.close()