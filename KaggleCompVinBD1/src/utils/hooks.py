from src.utils.timer import Timer
from src.models.model import BaseModel
from src.visualizations.visualization import Visualization
from src import hooks_log

from typing import Optional, List


# Following structure found in DetectronModeling, some modifications and simplifications
# https://github.com/facebookresearch/detectron2/blob/4aca4bdaa9ad48b8e91d7520e0d0815bb8ca0fb1/detectron2/engine/train_loop.py#L18
class HookBase:

    def __init__(self):
        from src.training_tasks.training_task import TrainingTask

        self.trainer: Optional[TrainingTask] = None
        pass

    def before_training(self):
        pass

    def after_training(self):
        pass

    def before_iteration(self):
        pass

    def after_iteration(self):
        pass


class StepTimer(HookBase):

    def __init__(self):
        super().__init__()

        self.training_timer: Timer = Timer()
        self.iteration_timer: Timer = Timer()

    def before_training(self):
        self.training_timer.start()

    def after_training(self):
        elapsed = self.training_timer.stop()
        hooks_log.info(f'Training elapsed: {elapsed:.8f}s')

    def before_iteration(self):
        self.iteration_timer.start()

    def after_iteration(self):
        elapsed = self.iteration_timer.stop()
        self.trainer.storage.put_item('iteration_time', elapsed)


class PeriodicStepFuncHook(HookBase):

    def __init__(self, frequency: int, func: callable, after_training: bool = True):
        super().__init__()

        self.frequency: int = frequency
        self.func: callable = func

        self.run_after_training: bool = after_training

    def __run_func__(self):
        results = self.func()

        if results:

            for key in results:
                self.trainer.storage.put_item(key, results[key])

    def after_iteration(self):
        if self.trainer.iter % self.frequency == 0:
            self.__run_func__()

    def after_training(self):
        if self.run_after_training:
            self.__run_func__()


class CheckpointHook(PeriodicStepFuncHook):

    def __init__(self, name: str, frequency: int):
        super().__init__(frequency, self.checkpoint, after_training=True)
        self.__static_name__ = name

    def checkpoint(self):
        assert self.trainer.__getattribute__('model') is not None, 'trainer has no model to checkpoint'

        model: BaseModel = self.trainer.model
        model.checkpoint(name=self.name, state=self.build_state())

    def build_state(self):
        # You can override this if checkpoints need to hold specific data.
        pass

    @property
    def name(self):
        # Overwrite this if you want it to be a name that can change based on iterations or something
        return self.__static_name__


class TrainingVisualizationHook(HookBase):

    def __init__(self):
        super().__init__()
        self.visualizer = Visualization()

    def after_iteration(self):
        loss_item: List[float] = self.trainer.storage.find_item("loss", 1)
        self.visualizer.add_data(loss_item)
