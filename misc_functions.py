import os

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
            try:
                index = file[:-4]
                indices.append(int(index))
            except:
                print('Invalid index: ' + file + '. Must be an integer.')

    indices.sort()
    return indices


# Parses a prediction text file in the static path and returns a list of lists
# of the form [prediction, probability].
def parse_predictions(path):
    predictions = []
    with open(path) as f:
        for line in f:
            to_list = line.strip().split()
            predictions.append(to_list)
    f.close()

    for prediction in predictions:
        prediction[0] = prediction[0].replace('_', ' ')

    return predictions


# Parses the layer info text file in the static path and returns a tuple of
# the information.
def parse_layer_info(path):
    info = []
    with open(path) as f:
        for line in f:
            try:
                info.append(eval(line.strip()))
            except:
                info.append(line.strip())
    f.close()
    return tuple(info)

# Runs the terminal command to execute the javascript file that updates the
# Data Observatory screens.
def load_to_gdo(label, layer):
    os.system("node ./load_to_gdo.js " + label + " " + layer)





