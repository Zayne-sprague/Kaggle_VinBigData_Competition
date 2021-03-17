
table_match = {
    'backbone.bottom_up.stem.conv1.weight': 'model.1.weight'
}

def resnet50_to_detectron(model_in, model_out):

    p2 = 'model.4.'
