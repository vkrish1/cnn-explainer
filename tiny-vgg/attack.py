import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import re
from shutil import copyfile
from glob import glob
from json import load, dump
from os.path import basename
from time import time

print(tf.__version__)


##################################################################
# First load, process, input images
##################################################################

SAVED_MODEL_LOCATION = "./trained_vgg_best.h5"
IMAGE_LOCATIONS = "../website_imgs/orig/*.jpeg" # otherwise './data/class_10_val/val_images/*.JPEG'
IMAGE_LABEL_MAP_LOCATION = "../website_imgs/website_imgs_class_dict_10.json"
# WARNING: WILL WRITE OUT ADV IMAGES HERE:
ADV_IMAGE_SAVE_LOCATION = "../website_imgs/blackbox"


def process_path_test(path):
    """
    Get the (class label, processed image) pair of the given image path. This
    funciton uses python primitives, so you need to use tf.py_funciton wrapper.
    This function uses global variables:

        WIDTH(int): the width of the targeting image
        HEIGHT(int): the height of the targeting iamge
        NUM_CLASS(int): number of classes

    The filepath encoding for test images is different from training images.

    Args:
        path(string): path to an image file
    """

    # Get the class
    path = path.numpy()
    image_name = basename(path.decode('ascii'))
    label_index = tiny_val_class_dict[image_name]['index']

    # Convert label to one-hot encoding
    label = tf.one_hot(indices=[label_index], depth=NUM_CLASS)
    label = tf.reshape(label, [NUM_CLASS])

    # Read image and convert the image to [0, 1] range 3d tensor
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [WIDTH, HEIGHT])

    return(img, label)


def prepare_for_training(dataset, batch_size=32, cache=True,
                         shuffle_buffer_size=1000):
    ' Batches, in case theres a ton of data '

    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()

    # Only shuffle elements in the buffer size
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Pre featch batches in the background
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


NUM_CLASS = 10
BATCH_SIZE = 32
LR = 0.001 # just to compile the loaded model
WIDTH = 64
HEIGHT = 64

# Create training and validation dataset
#tiny_val_class_dict = load(open('./data/val_class_dict_10.json', 'r'))
tiny_val_class_dict = load(open(IMAGE_LABEL_MAP_LOCATION, 'r'))

vali_images = IMAGE_LOCATIONS

# Create vali dataset
vali_path_dataset = tf.data.Dataset.list_files(vali_images)

vali_labeld_dataset = vali_path_dataset.map(
    lambda path: tf.py_function(
        process_path_test,
        [path],
        [tf.float32, tf.float32]
    )
)

vali_dataset = prepare_for_training(vali_labeld_dataset,
                                    batch_size=BATCH_SIZE)

# Get the first batch of data
for images, labels in vali_dataset.take(1):
    x_test = images.numpy()
    y_test = labels.numpy()


##################################################################
# Start of attack
##################################################################

from PIL import Image
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

# ART clf requires disabled:
tf.compat.v1.disable_eager_execution()

# Load the sbest saved model:
tiny_vgg = tf.keras.models.load_model(SAVED_MODEL_LOCATION, compile=False)

# Then compile with an optimizer
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
tiny_vgg.compile(optimizer=optimizer, loss=loss_object)


classifier = KerasClassifier(model=tiny_vgg, 
    clip_values=(0, 1), 
    use_logits=False)


attack = ProjectedGradientDescent(estimator=classifier,
    eps=16/255, eps_step=1/255, norm="inf", max_iter=200)

#attack = CarliniLInfMethod(classifier,
#    confidence=0.8, targeted=False, learning_rate=0.001)

x_test_adv = attack.generate(x=x_test)
outputs = classifier.predict(x_test_adv)

preds = np.argmax(outputs, axis=1)
trues = np.argmax(y_test, axis=1)

accuracy = np.sum(preds == trues) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
print("Ixs that worked: ")
print(np.where(preds != trues))

## Save a few of x_test_adv, please work:
for i in range(len(preds)):
    x = (x_test_adv[i]*255).astype(np.uint8)
    im = Image.fromarray(x)
    im.save(ADV_IMAGE_SAVE_LOCATION + '/x_adv_'+str(i)+'.jpeg')