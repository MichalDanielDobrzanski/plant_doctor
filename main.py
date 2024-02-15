# https://www.kaggle.com/code/piyushmishra1999/plant-disease-detection-using-fast-ai/notebook

# input dataset:
# https://data.mendeley.com/datasets/tywbtsjrjv/1

# use dataset with augmentation

from fastai import *
from fastai.vision.all import *
import os
from os import listdir

print(f"Is CUDA available: {torch.cuda.is_available()}")

def get_dir():
    path = os.path.join(os.getcwd(), os.path.join("input", "Plant_leave_diseases_dataset_with_augmentation"))
    return path

def get_label(file_path):
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_length = len(split_dir_name)
    label = split_dir_name[dir_length - 1]
    return label

files = get_image_files(get_dir())
print(f'File count: {len(files)}')

defaults.device = torch.device('cpu')

# batch_tfms = [Normalize()]
# batch_tfms = [Resize(256), Normalize()]
batch_tfms = [Resize(256)]


dls = ImageDataLoaders.from_name_func(get_dir(), files, label_func=get_label, bs=64, item_tfms=batch_tfms,device=torch.device('cpu'))
dls.show_batch()


from matplotlib import pyplot
pyplot.show()
