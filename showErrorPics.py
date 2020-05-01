import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

error_pics = np.load('error_pics.npy')
#print(error_pics)

for i in range(error_pics.shape[0]):
    arr = error_pics[i,:,:,:]
    print(arr.shape)
    arr=arr.reshape((28,28))
    print(arr)
    img = Image.fromarray(arr)
    #img.save('./error_images/out'+str(i)+'.jpeg')
    plt.imshow(arr, cmap='gray')
    plt.savefig('./error_images/out'+str(i)+'.jpeg')
    