# organize data and label
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import scipy.io as sio

data_dir = '/home/zkk/medical/cell/training/'
imgs = []
for i in range(1, 722, 1):

    a=i+10000
    print(a)
    b=str(a)
    c=b[2:5]
    print(c)
    file_path = data_dir + c+ '.png'
    #file_path = data_dir + '/{}.png'.format(i)
    img = img_to_array(load_img(file_path, grayscale=False, target_size=(224,224)))
    imgs.append(img)

imgs = np.asarray(imgs, 'float32')

np.save('images_dataset2_train4.npy', imgs)

print(imgs.shape)

# deal labels
"""
data = sio.loadmat('')
labels = [label[0] for label in data['img_label']]
print(labels)
np.save('labels_dataset2_train4.npy', labels)
"""