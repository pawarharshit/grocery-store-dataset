import os
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from grocery import Grocery

builder = Grocery()

#first time remove this comment 
# dl_config = tfds.download.DownloadConfig(register_checksums=True)
# builder.download_and_prepare(download_config=dl_config)

# and add comment in below line
builder.download_and_prepare()

train_dataset = builder.as_dataset(split="train")
validation_dataset = builder.as_dataset(split="test")

assert(isinstance(train_dataset, tf.data.Dataset))
assert(isinstance(validation_dataset, tf.data.Dataset))

for a_train_example in train_dataset.take(5):
    image,label = a_train_example["image"],a_train_example["label"]
    print(f"Image Shape : {image.shape}")
    print(f"Label : {label.numpy()}")
    print(f"Id : {id.numpy()}")
    print("--------------------------------------------")