from pathlib import Path
import numpy
import imageio
from skimage.transform import resize
import numpy as np
import dlib
import os
from multiprocessing import Pool


class process_img(object):
    # Process an image, applying any transformations necessary including
    # boundaries, clipping and resizing it
    def pre_process(self, image, return_boundary=False):
        boundary = self.detector(image, 2)
        if(len(boundary) == 0):
            return None
        else:
            aligned = self.clip_image(image, boundary[0])
            resized = resize(
                aligned, [self.required_size, self.required_size, 3])
            if return_boundary:
                return resized, boundary
            return resized

    def clip_image(self, image, boundary):
        top = np.clip(boundary.top(), 0, np.Inf).astype(np.int16)
        bottom = np.clip(boundary.bottom(), 0, np.Inf).astype(np.int16)
        left = np.clip(boundary.left(), 0, np.Inf).astype(np.int16)
        right = np.clip(boundary.right(), 0, np.Inf).astype(np.int16)
        return image[top:bottom, left:right]

    def __init__(self, required_size):
        self.required_size = required_size
        self.detector = dlib.get_frontal_face_detector()


class generate_image(process_img):
    # Take a dataset and process it and save it in a new path.
    def __init__(self, original_path, new_path, required_size=96,
                 as_npy=False):
        super().__init__(required_size)
        self.original_path = Path(original_path)
        self.new_path = Path(new_path)
        self.as_npy = as_npy

    def get_folder_list(self):
        self.file_dir = [x for x in self.original_path.iterdir() if x.is_dir()]

    def create_directory(self, path):
        if not path.is_dir():
            os.makedirs(path)

    def create_dataset(self, directory):
        new_directory = self.new_path/directory.parts[-1]
        self.create_directory(new_directory)
        self.generate_new_images(directory, new_directory)

    def generate(self):
        self.get_folder_list()
        pool = Pool(os.cpu_count())
        pool.map(self.create_dataset, self.file_dir)

    def generate_new_images(self, load_path, save_path):
        if self.as_npy:
            temp = list()
            for images in list(load_path.glob('*.jpg')):
                modified_image = self.pre_process(imageio.imread(images))
                if modified_image is not None:
                    temp.append((255 * modified_image).astype(np.uint8))
            np.save(save_path, np.array(temp))
        else:
            for images in list(load_path.glob('*.jpg')):
                modified_image = self.pre_process(imageio.imread(images))
                if modified_image is not None:
                    imageio.imsave(
                        save_path/images.parts[-1],
                        (255 * modified_image).astype(np.uint8))
if __name__ =="__main__":
    gen = generate_image("/D/work/ML/Faces/lfw-deepfunneled/",
                         new_path="/home/touchdown/lfw-preprocessed")
    gen.generate()

