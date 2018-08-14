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
    input_image = '/static/images/' + label + '.jpg'

    # Return to index page.
    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('selected.html', input_image=input_image,
                           label=label, form=form)


@app.route('/classified/<label>', methods=['GET', 'POST'])
def classified(label):
    # Path to selected input image.
    input_image = './app/static/images/' + label + '.jpg'

    # Compute top 5 predictions.
    predictions = predict(input_image)

    # Generate heatmap and Grad-CAM for image.
    heatmap, gradcam = generate_gradcam(input_image)

    # Return to index page.
    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('classified.html', predictions=predictions,
                           heatmap=heatmap, gradcam=gradcam, label=label, form=form)


@app.route('/<label>/<block>', methods=['GET', 'POST'])
def visualize_block_filters(label, block):
    path_to_image = './app/static/images/' + label + '.jpg'
    filter_outputs = visualize_filter_outputs(path_to_image, block)

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


