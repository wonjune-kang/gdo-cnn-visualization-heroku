from flask import render_template, redirect, url_for, request
from app import app
from app.forms import SelectionForm, ReturnForm
import os


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = SelectionForm()
    if form.validate_on_submit():
        if form.view.data:
            return redirect(url_for('selected', label=form.select_image.data))
        elif form.classify.data:
            return redirect(url_for('classified', label=form.select_image.data))
        elif form.visualize_filters.data:
            return redirect(url_for('visualize_layer_filters',
                                    label=form.select_image.data,
                                    layer=form.select_layer.data))
        elif form.layer_info.data:
            return redirect(url_for('layer_info', layer=form.select_layer.data))

    return render_template('index.html', form=form)


@app.route('/input/<label>', methods=['GET', 'POST'])
def selected(label):
    # Path to selected input image.
    path_to_input = '/static/images/' + label + '.jpg'

    # Return to index page.
    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('selected.html', input_image=path_to_input,
                           label=label, form=form)


@app.route('/classified/<label>', methods=['GET', 'POST'])
def classified(label):
    # Paths to selected input image, heatmap, and guided Grad-CAM.
    path_to_heatmap = '/static/cam_heatmaps/' + label + '.png'
    path_to_guided = '/static/cam_guided/' + label + '.png'

    # Get the top 5 predictions.
    # predictions = parse_predictions('./app/static/predictions/' + label + '.txt')
    predictions = None
    # Return to index page.
    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('classified.html', predictions=predictions,
                           heatmap=path_to_heatmap, guided=path_to_guided,
                           label=label, form=form)


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

@app.route('/<label>/<layer>', methods=['GET', 'POST'])
def visualize_layer_filters(label, layer):
    path_to_visualizations = '/static/filter_visualizations/' + layer + '/'
    path_to_outputs = '/static/filter_outputs/' + label + '/' + layer + '/'
    viz_and_outputs = []

    filter_indices = []
    for file in os.listdir('./app/'+path_to_visualizations):
        if file[0] != '.':
            try:
                index = file[:-4]
                filter_indices.append(int(index))
            except:
                print('Invalid index: ' + file + '. Must be an integer.')

    filter_indices.sort()

    for idx in filter_indices:
        visualization = path_to_visualizations + '%d.png' % idx
        output = path_to_outputs + '%d.png' % idx
        viz_and_outputs.append((visualization, output))

    # Return to index page.
    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('visualize_all_filters.html',
                           viz_and_outputs=viz_and_outputs, form=form)


@app.route('/info/<layer>', methods=['GET', 'POST'])
def layer_info(layer):
    path_to_info = '/static/layer_info/' + layer + '.txt'

    print("Current working directory: " + os.getcwd())
    info = []
    with open(path_to_info) as f:
        for line in f:
            try:
                info.append(eval(line.strip()))
            except:
                info.append(line.strip())
    f.close()

    name, activation, num_filters, dims, strides = tuple(info)

    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('layer_info.html', name=name, activation=activation,
                           num_filters=num_filters, dims=dims, strides=strides,
                           form=form)


def parse_predictions(path):
    predictions = []
    with open(path) as f:
        for line in f:
            to_list = tuple(line.strip().split())
            predictions.append(to_list)
    
    f.close()
    return predictions




