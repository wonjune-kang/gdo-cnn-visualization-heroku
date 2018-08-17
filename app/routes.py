from flask import render_template, redirect, url_for, request
from app import app
from app.forms import SelectionForm, ReturnForm
from misc_functions import get_filter_indices, parse_predictions, parse_layer_info


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

    print(form.errors)
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
    predictions = parse_predictions('./app/static/predictions/' + label + '.txt')

    # Return to index page.
    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('classified.html', predictions=predictions,
                           heatmap=path_to_heatmap, guided=path_to_guided,
                           label=label, form=form)


@app.route('/<label>/<layer>', methods=['GET', 'POST'])
def visualize_layer_filters(label, layer):
    path_to_visualizations = '/static/filter_visualizations/' + layer + '/'
    path_to_outputs = '/static/filter_outputs/' + label + '/' + layer + '/'
    
    viz_and_outputs = []
    filter_indices = get_filter_indices('./app/'+path_to_visualizations)
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
    path_to_info = './app/static/layer_info/' + layer + '.txt'
    name, activation, num_filters, dims, strides = parse_layer_info(path_to_info)

    form = ReturnForm()
    if form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('layer_info.html', name=name, activation=activation,
                           num_filters=num_filters, dims=dims, strides=strides,
                           form=form)


