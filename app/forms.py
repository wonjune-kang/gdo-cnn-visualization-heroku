from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
from wtforms.validators import DataRequired
from misc_functions import create_image_choices


class SelectionForm(FlaskForm):
    select_image = SelectField('Select Image', validators=[DataRequired()],
                               choices=create_image_choices('./app/static/images/'))

    select_layer = SelectField('Select Layer', validators=[DataRequired()],
                               choices=[('block1_conv1', 'block1_conv1'),
                                        ('block2_conv1', 'block2_conv1'),
                                        ('block3_conv1', 'block3_conv1'),
                                        ('block4_conv1', 'block4_conv1'),
                                        ('block5_conv1', 'block5_conv1')])

    view = SubmitField("Visualize")