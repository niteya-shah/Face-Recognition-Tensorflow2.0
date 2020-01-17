from pathlib import Path
import numpy
import imageio
import random
from skimage.transform import resize
from skimage.io import imshow
import numpy as np
import dlib
import os
from numba import jit

class image_generator:

    def image_data(self):
        return_list = np.zeros([self.number_of_images,3,self.required_size,self.required_size,3], dtype = np.float32)
        file_dir = [x for x in self.path.iterdir() if x.is_dir()]
        random.shuffle(file_dir)
        for people in range(self.number_of_images):
            positive = random.randint(0,len(file_dir) - 1)
            negative = random.randint(0,len(file_dir) - 1)
            image_list = [imageio.imread(images) for images in random.choices(list(file_dir[positive].glob('*.jpg')), k=2)]
            image_list.append(imageio.imread(random.choice(list(file_dir[negative].glob('*.jpg')))))
            image_list = np.array(image_list, dtype = np.float32)
            return_list[people] = image_list
        return np.split(return_list, 3,axis = 1)

    def __init__(self, path, number_of_images = 1000, required_size = 96):
        self.path = Path(path)
        self.number_of_images = number_of_images
        self.required_size = required_size
