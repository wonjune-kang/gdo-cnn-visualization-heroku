from flask import render_template, redirect, url_for, request
from app import app
from app.forms import SelectionForm
from rq import Queue
from worker import conn
from misc_functions import parse_predictions, parse_layer_info, load_to_gdo


# Main controller page for the Data Observatory visualizations.
# User selects the image and layer to visualize.
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = SelectionForm()
    # label = form.select_image.data
    # layer = form.select_layer.data
    # load_to_gdo(label, layer)

    if form.validate_on_submit():
        label = form.select_image.data
        layer = form.select_layer.data
        q = Queue(connection=conn)
        q.enqueue(load_to_gdo, label, layer)
        # load_to_gdo(label, layer)

    return render_template('index.html', form=form)


# Shows the title page.
@app.route('/title', methods=['GET', 'POST'])
def title():
    network = 'VGG16'
    return render_template('title.html', network=network)


# Shows the input image.
@app.route('/input/<label>', methods=['GET', 'POST'])
def input(label):
    return render_template('input.html', label=label)


# Shows the network architecture image.
@app.route('/network-structure', methods=['GET', 'POST'])
def network_structure():
    return render_template('network_structure.html')


# Shows information about the selected layer.
@app.route('/info/<layer>', methods=['GET', 'POST'])
def layer_info(layer):
    # Parse the layer specifications from the information text file.
    path_to_info = './app/static/layer_info/' + layer + '.txt'
    name, activation, num_filters, dims, strides = parse_layer_info(path_to_info)

    dims_to_str = str(dims[0]) + ' x ' + str(dims[1]) + ' x ' + str(dims[2])
    strides_to_str = str(strides[0]) + ' x ' + str(strides[1])

    return render_template('layer_info.html', name=name, activation=activation,
                           num_filters=num_filters, dims=dims_to_str,
                           strides=strides_to_str)


# Shows the top 5 predictions for the selected image.
@app.route('/predictions/<label>', methods=['GET', 'POST'])
def predictions(label):
    # Parse the top 5 predictions from the predictions text file.
    predictions = parse_predictions('./app/static/predictions/'+label+'.txt')

    return render_template('predictions.html', predictions=predictions)


# Shows the heatmap and guided Grad-CAM for the selected image.
@app.route('/grad-cam/<label>', methods=['GET', 'POST'])
def grad_cam(label):
    # Parse the top 5 predictions from the predictions text file.
    return render_template('grad_cam.html', label=label)


# Shows a filter from the selected layer and its output.
@app.route('/visualize/<label>/<layer>/<filter_idx>', methods=['GET', 'POST'])
def visualize_single_filter(label, layer, filter_idx):
    return render_template('visualize_single_filter.html', label=label,
                           layer=layer, filter_idx=filter_idx)





