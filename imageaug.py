import keras
import cv2
import os
import glob
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


Input_folder = "/content/drive/My Drive/protomask/observations/experiements/data/without_mask" # Enter Directory of all images
Output_folder = "/content/drive/My Drive/protomask/training/augmentation/without_mask"

#Data augmentation parameters
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


data_path = os.path.join(Input_folder,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    i = 0
    for batch in datagen.flow (x, batch_size=1, save_to_dir =Output_folder,save_prefix="aug",save_format='jpg'):
      i+= 1
      if i > 36:
        break
