from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
from wtforms.validators import DataRequired
from misc_functions import create_image_choices


class SelectionForm(FlaskForm):
    select_image = SelectField('Select Image', validators=[DataRequired()],
                               choices=create_image_choices('./app/static/images/'))

    select_block = SelectField('Select Block', validators=[DataRequired()],
                                               choices=[('block1', 'block 1'),
                                                        ('block2', 'block 2'),
                                                        ('block3', 'block 3'),
                                                        ('block4', 'block 4'),
                                                        ('block5', 'block 5')])

    view = SubmitField("View Selected Image")
    classify = SubmitField("Classify Selected Image")
    visualize_filters = SubmitField("Visualize Filters for Block")
    layer_info = SubmitField('Block Info')


class ReturnForm(FlaskForm):
    return_home = SubmitField('Back')