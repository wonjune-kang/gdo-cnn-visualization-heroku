import keras
from keras import backend as K
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.layers.core import Lambda
from keras.models import Model
from keras.activations import relu
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import sys
import os


# Define VGG16 model and default TensorFlow graph.
model = vgg16.VGG16(weights='imagenet')
graph = tf.get_default_graph()
print('Model loaded.')


# Scales an image such that its shorter side is equal to new_side, then takes
# the center crop. Default value of new_side is 224 px for ImageNet.
def crop_and_resize(image, new_side=224):
    width, height = image.size
    shorter = min(width, height)

    left = (width - shorter)/2
    top = (height - shorter)/2
    right = (width + shorter)/2
    bottom = (height + shorter)/2

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((new_side, new_side), Image.ANTIALIAS)


# Returns a scaled, cropped, and preprocessed image for ImageNet CNNs.
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
    preprocessed = preprocess_input(image_batch.copy())
    return preprocessed


# Writes the top 5 predictions for the new image to the static directory.
def write_vgg16_predictions(path_to_image, label):
    def predict(path_to_image):
        image = load_image(path_to_image)
         
        # Get the top 5 predictions for each class and decode labels.
        predictions = model.predict(image)
        labels = decode_predictions(predictions)

        return(labels)

    top5 = predict(path_to_image)[0]

    with open('./app/static/predictions/'+label+'.txt', 'w') as f:
        for _, prediction, prob in top5:
            percentage = ('%.2f' % (prob*100)) + '%'
            print(prediction+'\t\t'+str(percentage), file=f)

    f.close()
    print("Finished writing predictions for " + label + ".")


# Generates all filter outputs for the new image and saves them to the
# static directory.
def generate_filter_outputs(path_to_image, label):
    def visualize_filter_output(label, layer_idx):
        layer = model.layers[layer_idx]
        image = load_image(path_to_image)

        # Placeholder for input image.
        input_image = model.input

        # TensorFlow function used to get the output from the input.
        functor = K.function([input_image, K.learning_phase()], [layer.output])

        layer_output = np.squeeze(functor([image, 0.0]))
        return layer_output


    # Dictionary mapping from VGG16 layer names to layer indices.
    block_layers = {
                    'block1_conv1': 1,
                    'block2_conv1': 4,
                    'block3_conv1': 7,
                    'block4_conv1': 11,
                    'block5_conv1': 15
                    }

    # Initialize dictionary mapping from layer names to the filter indices to
    # be processed.
    layer_filter_indices = {
                            'block1_conv1': [],
                            'block2_conv1': [],
                            'block3_conv1': [],
                            'block4_conv1': [],
                            'block5_conv1': []
                            }

    # Add filter indices to layer_filter_indices.
    for layer in layer_filter_indices.keys():
        for idx in os.listdir('./app/static/filter_visualizations/'+layer):
            if not idx.startswith('.'):
                layer_filter_indices[layer].append(int(idx[:-4]))

    # Iterate through all layers to consider.
    for layer, indices in layer_filter_indices.items():
        layer_idx = block_layers[layer]

        # Get all layer outputs.
        layer_output = visualize_filter_output(label, layer_idx)

        # Save output for filter indices in layer.
        for filter_idx in indices:
            filtered = layer_output[:,:,filter_idx]
            rescaled = (255.0 / filtered.max() * (filtered - filtered.min())).astype(np.uint8)
            im = Image.fromarray(rescaled)

            # Save image to path.
            path_to_save = './app/static/filter_outputs/'+label+'/'+layer
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            im.save(path_to_save+'/'+str(filter_idx)+'.png')

        print("Finished saving filter outputs for " + layer + ".")
    print("Finished saving all filter outputs for " + label + ".")


# Generates the Gradient-weighted Class Activation Mapping (Grad-CAM) for the
# new image and saves the heatmap and guided Grad-CAM to the static directory.
def generate_gradcam(path_to_image, label):
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

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(image)
        cam = 255 * cam / np.max(cam)
        return np.uint8(cam), heatmap

    preprocessed_input = load_image(path_to_image)

    # Feed image into model and get top prediction.
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions)

    # Generate Grad-CAM and heatmap.
    cam, heatmap = \
        grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")

    # Write heatmap to static directory.
    cv2.imwrite('./app/static/cam_heatmaps/'+label+'.png', cam)
    print("Grad-CAM heatmap generated for " + label + ".")

    # Generate guided Grad-CAM and save to static directory.
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([preprocessed_input, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    cv2.imwrite('./app/static/cam_guided/'+label+'.png', deprocess_image(gradcam))
    print("Guided Grad-CAM generated for " + label + ".")


if __name__ == '__main__':
    # for image in os.listdir('./app/static/images'):
    #     if image[0] != '.':
    #         generate_gradcam('./app/static/images/'+image, image[:-4])

    path_to_image = './app/static/images/ball.jpg'
    label = 'ball'

    write_vgg16_predictions(path_to_image, label)
    generate_filter_outputs(path_to_image, label)
    # generate_gradcam(path_to_image, label)

