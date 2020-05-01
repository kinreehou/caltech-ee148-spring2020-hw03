import heapq
import numpy as np 
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

results = np.load('intermediateResult.npy')
targets = np.load('targets.npy')
results = results.reshape((10000,120))
targets = targets.reshape((10000,))

print(results.shape)
print(targets.shape)
#for i in range(results.shape[0]):

'''
X = results
X_embedded = TSNE(n_components=2).fit_transform(X)

print(X_embedded)
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=targets.astype(float))
plt.savefig('TSNE_visualization.jpg')
plt.show()
'''

image_idx = 1230
v1 = results[image_idx,:]
h = []
heapq.heapify(h)

for i in range(10000):
    v2 = results[i,:]
    dist = np.linalg.norm(v2-v1)
    if len(h)<12:
        heapq.heappush(h, (-dist,i))
    else:
        if -dist>h[0][0]:
            heapq.heappushpop(h, (-dist,i))
h = sorted(h, reverse=True)
print(h)

test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))



for i in range(9):
    pic_idx = h[i][1]
    arr = test_dataset[pic_idx][0].numpy()
    arr = arr.reshape((28,28))
    img = Image.fromarray(arr)
    plt.imshow(arr, cmap='gray')
    plt.savefig('./nearest_pics/I1230-'+str(i)+'.jpeg')




  