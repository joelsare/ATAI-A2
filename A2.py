# FROM https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency


from tensorflow.keras.applications.vgg16 import VGG16 as Model

model = Model(weights='imagenet', include_top=True)
# model.summary()

replace2linear = ReplaceToLinear()

score = CategoricalScore([151, 987, 985])

# Image titles
image_titles = ['Chihuahua', 'corn', 'daisy']

# Load images and Convert them to a Numpy array
img1 = load_img('images/Chihuahua.jpg', target_size=(224, 224))
img2 = load_img('images/corn.jpg', target_size=(224, 224))
img3 = load_img('images/daisy.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data for VGG16
X = preprocess_input(images)

# Rendering
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
plt.savefig("img.jpg")

saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)

# Generate saliency map

# Vanilla
# saliency_map = saliency(score, X)

# SmoothGrad
saliency_map = saliency(score,
                        X,
                        smooth_samples=20, # The number of calculating gradients iterations.
                        smooth_noise=0.20) # noise spread level.

## Since v0.6.0, calling `normalize()` is NOT necessary.
# saliency_map = normalize(saliency_map)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(saliency_map[i], cmap='jet')
    ax[i].axis('off')
plt.tight_layout()
plt.savefig("smoothgrad.jpg")