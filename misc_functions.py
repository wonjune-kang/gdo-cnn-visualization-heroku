from PIL import Image
import base64
import os


# Scales an image such that its shorter side is equal to new_side, then takes
# the center crop. Default value of new_side is 224 for ImageNet preprocessing.
def crop_and_resize(image, new_side=224):
    width, height = image.size
    shorter = min(width, height)

    left = (width - shorter)/2
    top = (height - shorter)/2
    right = (width + shorter)/2
    bottom = (height + shorter)/2

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((new_side, new_side), Image.ANTIALIAS)


# Given a path to a directory with images with extensions (.jpg, .png, etc.),
# returns a sorted list of image choices for a WTForm SelectForm.
def create_image_choices(path):
    choices = []
    for choice in os.listdir(path):
        if choice[0] != '.':
            file = choice[:-4]
            choices.append((file, file))

    choices.sort()
    return choices


# Given a path to a directory with filter visualizations for a VGG16 block,
# returns a list of filter indices to be processed for the layer of that block.
def get_filter_indices(path):
    indices = []
    for file in os.listdir(path):
        if file[0] != '.':
            index = file[:-4]
            indices.append(int(index))

    indices.sort()
    return indices


