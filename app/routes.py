from flask import render_template, redirect, url_for, request
from app import app
from app.forms import SelectionForm, ReturnForm
from vgg16 import *


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
            return redirect(url_for('visualize_block_filters',
                                    label=form.select_image.data,
                                    block=form.select_block.data))
        elif form.layer_info.data:
            return redirect(url_for('layer_info', block=form.select_block.data))

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
    path_to_input = './app/static/images/' + label + '.jpg'
    path_to_heatmap = '/static/cam_heatmaps/' + label + '.png'
    path_to_guided = '/static/cam_guided/' + label + '.png'

    # Compute top 5 predictions.
    predictions = predict(path_to_input)

    # Return to index page.
    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('classified.html', predictions=predictions,
                           heatmap=path_to_heatmap, guided=path_to_guided,
                           label=label, form=form)


@app.route('/<label>/<block>', methods=['GET', 'POST'])
def visualize_block_filters(label, block):
    path_to_input = './app/static/images/' + label + '.jpg'
    filter_outputs = visualize_filter_outputs(path_to_input, block)

    # Return to index page.
    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('visualize_all_filters.html',
                           filter_outputs=filter_outputs, form=form)


@app.route('/info/<block>', methods=['GET', 'POST'])
def layer_info(block):
    name, activation, num_filters, dims, strides = get_layer_info(block)

    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('layer_info.html', name=name, activation=activation,
                           num_filters=num_filters, dims=dims, strides=strides,
                           form=form)





