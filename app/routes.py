from flask import render_template, redirect, url_for, request
from app import app
from app.forms import SelectionForm
from misc_functions import get_filter_indices, parse_predictions, \
                           parse_layer_info, update_screens


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = SelectionForm()
    if form.validate_on_submit():
        label = form.select_image.data
        layer = form.select_layer.data
        update_screens(label, layer)

    return render_template('index.html', form=form)


@app.route('/title', methods=['GET', 'POST'])
def title():
    return render_template('title.html')


@app.route('/input/<label>', methods=['GET', 'POST'])
def input(label):
    return render_template('input.html', label=label)


@app.route('/network-structure', methods=['GET', 'POST'])
def network_structure():
    return render_template('network_structure.html')


@app.route('/info/<layer>', methods=['GET', 'POST'])
def layer_info(layer):
    # Parse the layer specifications from the information text file.
    path_to_info = './app/static/layer_info/' + layer + '.txt'
    name, activation, num_filters, dims, strides = parse_layer_info(path_to_info)

    return render_template('layer_info.html', name=name, activation=activation,
                           num_filters=num_filters, dims=dims, strides=strides)


@app.route('/predictions/<label>', methods=['GET', 'POST'])
def predictions(label):
    # Parse the top 5 predictions from the predictions text file.
    predictions = parse_predictions('./app/static/predictions/'+label+'.txt')

    return render_template('predictions.html', predictions=predictions)


@app.route('/grad-cam/<label>', methods=['GET', 'POST'])
def grad_cam(label):
    # Parse the top 5 predictions from the predictions text file.
    return render_template('grad_cam.html', label=label)


@app.route('/visualize/<label>/<layer>/<filter_idx>', methods=['GET', 'POST'])
def visualize_single_filter(label, layer, filter_idx):
    # Get the filter indices to access from the ones available in the layer.
    path_to_visualizations = './app/static/filter_visualizations/' + layer + '/'
    filter_indices = get_filter_indices(path_to_visualizations)

    return render_template('visualize_single_filter.html', label=label,
                           layer=layer, filter_idx=filter_idx)



