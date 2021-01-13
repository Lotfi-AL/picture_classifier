import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import csv

import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import shutil

import os


root_dir = "./pic"


def init_classes():
    df = pd.read_csv("train_info.csv", sep=",", header=0)
    columns_of_interest = ['genre']
    df = df[columns_of_interest]
    df.dropna(how='any', inplace=True)
    print(df)
    pic_classes = set()

    for x in df["genre"]:
        pic_classes.add(x)

    print(pic_classes)
    with open("all_classes.txt", "w") as f:
        f.write("\n".join(pic_classes))


def init_dirs():
    with open("all_classes.txt", "r") as f:
        for item in f:
            os.mkdir("./pic/"+item.strip())


def move_images():
    df = pd.read_csv("train_info.csv", sep=",", header=0)
    columns_of_interest = ["filename", "genre"]
    df = df[columns_of_interest]
    df.dropna(how='any', inplace=True)
    # mov("abstract", "1.jpg")
    result = [mov(genre, filename)
              for genre, filename in zip(df['genre'], df['filename'])]


def test_mov(genre, filename):
    source = './test_folder/'+filename
    root_dest = './pic/'
    dest1 = root_dest+genre
    print(source)
    print(dest1)
    shutil.move(source, dest1)


def mov(genre, filename):
    print(genre, filename)
    source = './train/train/'+filename
    # source = os.path.join("./train/", filename)
    print(source)
    # source = "./train/"+filename
    root_dest = './pic/'
    dest1 = root_dest+genre
    shutil.move(source, dest1)


move_images()
