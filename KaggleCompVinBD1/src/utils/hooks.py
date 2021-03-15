
# Following structure found in Detectron2, some modifications and simplifications
# https://github.com/facebookresearch/detectron2/blob/4aca4bdaa9ad48b8e91d7520e0d0815bb8ca0fb1/detectron2/engine/train_loop.py#L18
class HookBase:

    def __init__(self):
        pass

    def before_training(self):
        pass

    def after_training(self):
        pass

    def before_iteration(self):
        pass

    def after_iteration(self):
        pass

