from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from misc_functions import crop_and_resize, get_filter_indices


# Define VGG16 model.
model = vgg16.VGG16(weights='imagenet')
graph = tf.get_default_graph()


# Given a block, returns the index of the first layer of that block in VGG16.
def block_to_layer(block):
    block2layer = {'block1': 1,
                   'block2': 4,
                   'block3': 7,
                   'block4': 11,
                   'block5': 15}

    return block2layer[block]


# Returns a scaled, cropped, and preprocessed image for VGG16.
def load_image(path_to_image):
    # Load the image in PIL format.
    original = load_img(path_to_image)

    # Rescale to 224 px on the shorter side and take the center crop.
    resized = crop_and_resize(original, new_side=224)

    # Convert to numpy array.
    image_np = img_to_array(resized)

    # Convert single image to batch format by adding an extra dimension.
    image_batch = np.expand_dims(image_np, axis=0)

    # Preprocess for VGG16 model.
    preprocessed = vgg16.preprocess_input(image_batch.copy())
    return preprocessed


# Gets the name, activation function, number of filters, filter dimensions,
# and filter strides of a layer in a block.
def get_layer_info(block):
    layer_idx = block_to_layer(block)
    layer = model.layers[layer_idx]
    weights = layer.get_weights()[0]

    name = layer.name
    activation = layer.activation.__name__
    num_filters = layer.filters
    dims = weights[:,:,:,0].shape
    strides = layer.strides

    return (name, activation, num_filters, dims, strides)


# Gets the top 5 predictions for an image.
def predict(path_to_image):
    with graph.as_default():
        preprocessed = load_image(path_to_image)

        # Get the predicted probabilities for each class and decode labels.
        predictions = model.predict(preprocessed)
        labels = vgg16.decode_predictions(predictions)
        
        # Return a list of tuples of readable labels and their probabilities.
        readable_labels = []
        for _, label, prob in labels[0]:
            new_label = label.replace('_', ' ')
            readable_labels.append((new_label, round(prob*100, 2)))

    tf.reset_default_graph()
    return readable_labels


# Feed an input image through the network and get the filter outputs of the
# layer corresponding to the specified block.
def visualize_filter_outputs(path_to_image, block):
    path_to_visualizations = '/static/filter_visualizations/' + block
    layer_idx = block_to_layer(block)

    with graph.as_default():
        layer = model.layers[layer_idx]
        preprocessed = load_image(path_to_image)

        # Placeholder for the input image.
        input_image = model.input

        # Create a TensorFlow function that is used to get the output given
        # the input.
        functor = K.function([input_image, K.learning_phase()], [layer.output])
        filter_outputs = np.squeeze(functor([preprocessed, 0.0]))

        # Get the outputs for the 32 filters in the layers with the highest
        # visualization activations, as given in path_to_visualizations.
        layer_outputs = []
        for i in get_filter_indices('./app/'+path_to_visualizations):
            # Path to the filter visualization image.
            filter_visualization = path_to_visualizations+'/%d.png' % i

            # Output at current filter index.
            filtered = filter_outputs[:,:,i]

            # Save filter output image in base64 format to be decoded in HTML.
            buffer = BytesIO()
            im = Image.fromarray(np.uint8(filtered*255))
            im.save(buffer, format='PNG')
            buffer.seek(0)
            data_uri = base64.b64encode(buffer.getvalue()).decode('ascii')
            layer_outputs.append((filter_visualization, data_uri))

    tf.reset_default_graph()
    return layer_outputs





