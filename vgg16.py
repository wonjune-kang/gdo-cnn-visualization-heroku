from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.activations import relu
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
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

        return layer_outputs


# Generates the Gradient-weighted Class Activation Mapping (Grad-CAM) of the
# image. Returns the heatmap and the guided Grad-CAM.
def generate_gradcam(img_path):

    # Target function to maximize the given category index.
    def target_category_loss(x, category_index, nb_classes):
        return tf.multiply(x, K.one_hot([category_index], nb_classes))

    def target_category_loss_out_shape(input_shape):
        return input_shape

    # Utility function to normalize a tensor by its L2 norm.
    def normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    def register_gradient():
        if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) * \
                    tf.cast(op.inputs[0] > 0., dtype)

    def compile_saliency_function(model, activation_layer='block5_conv3'):
        input_img = model.input
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
        layer_output = layer_dict[activation_layer].output
        max_output = K.max(layer_output, axis=3)
        saliency = K.gradients(K.sum(max_output), input_img)[0]
        return K.function([input_img, K.learning_phase()], [saliency])

    def modify_backprop(model, name):
        with graph.gradient_override_map({'Relu': name}):
            # Get layers that have an activation function.
            layer_dict = [layer for layer in model.layers[1:]
                          if hasattr(layer, 'activation')]

            # Replace ReLU activation.
            for layer in layer_dict:
                if layer.activation == relu:
                    layer.activation = tf.nn.relu

            # Re-instantiate a new model.
            new_model = vgg16.VGG16(weights='imagenet')
        return new_model

    def deprocess_image(x):
        if np.ndim(x) > 3:
            x = np.squeeze(x)

        # Normalize tensor: center on 0., ensure std dev is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # Clip to [0, 1].
        x += 0.5
        x = np.clip(x, 0, 1)

        # Convert to RGB array.
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # Computes the Grad-CAM outputs.
    def grad_cam(input_model, image, category_index, layer_name):
        num_classes = 1000
        target_layer = \
                lambda x: target_category_loss(x, category_index, num_classes)

        x = input_model.layers[-1].output
        x = Lambda(target_layer, output_shape=target_category_loss_out_shape)(x)
        model = Model(input_model.layers[0].input, x)

        loss = K.sum(model.layers[-1].output)
        conv_output = \
                [l for l in model.layers if l.name is layer_name][0].output

        # Compute the gradients of the target function with respect to the
        # convolutional layer outputs using backpropagation.
        grads = normalize(K.gradients(loss, conv_output)[0])
        gradient_function = \
                K.function([model.layers[0].input], [conv_output, grads])

        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis = (0, 1))
        cam = np.ones(output.shape[0 : 2], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)

        # Return to BGR [0, 255] from the preprocessed image.
        image = image[0, :]
        image -= np.min(image)
        image = np.minimum(image, 255)

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_HSV)
        cam = np.float32(cam) + np.float32(image)
        cam = 255 * cam / np.max(cam)
        return cam, heatmap

    with graph.as_default():
        preprocessed_input = load_image(img_path)

        predictions = model.predict(preprocessed_input)
        predicted_class = np.argmax(predictions)
        cam, heatmap = \
            grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")

        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp')
        saliency_fn = compile_saliency_function(guided_model)
        saliency = saliency_fn([preprocessed_input, 0])
        gradcam = saliency[0] * heatmap[..., np.newaxis]
        gradcam = deprocess_image(gradcam)

        # Save heatmap image in base64 format to be decoded in HTML.
        heatmap_buffer = BytesIO()
        heatmap_img = Image.fromarray(np.uint8(cam))
        heatmap_img.save(heatmap_buffer, format='PNG')
        heatmap_buffer.seek(0)
        heatmap_data_uri = \
                base64.b64encode(heatmap_buffer.getvalue()).decode('ascii')

        # Save guided Grad-CAM image in base64 format to be decoded in HTML.
        guided_buffer = BytesIO()
        guided_img = Image.fromarray(np.uint8(gradcam))
        guided_img.save(guided_buffer, format='PNG')
        guided_buffer.seek(0)
        guided_data_uri = \
                base64.b64encode(guided_buffer.getvalue()).decode('ascii')

        return heatmap_data_uri, guided_data_uri


