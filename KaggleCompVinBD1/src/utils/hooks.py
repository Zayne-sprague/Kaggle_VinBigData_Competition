from torch import optim
from typing import Optional, List

from src.utils.timer import Timer
from src.modeling.models.model import BaseModel
from src.visualizations.visualization import Visualization
from src import hooks_log, training_log



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

    def on_resume(self):
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


def periodic(f):
    def decor(self):
        assert isinstance(self, PeriodicStepHook), "can only use the periodic decorators n periodic hooks"

        if self.trainer.iter > self.trainer.start_iter and self.trainer.iter % self.frequency == 0:
            return f(self)
        return None

    return decor


class PeriodicStepHook(HookBase):

    def __init__(self, frequency: int):
        super().__init__()

        self.frequency: int = frequency


class PeriodicStepFuncHook(PeriodicStepHook):

    def __init__(self, frequency: int, func: callable, after_training: bool = True):
        super().__init__(frequency)

        self.func: callable = func

        self.run_after_training: bool = after_training

    def __run_func__(self):
        results = self.func()

        if results:

            for key in results:
                self.trainer.storage.put_item(key, results[key])

    @periodic
    def after_iteration(self):
        self.__run_func__()

    def after_training(self):
        if self.run_after_training:
            self.__run_func__()


class CheckpointHook(PeriodicStepFuncHook):

    def __init__(self, frequency: int, name: str, permanent_checkpoints: int = 0, keep_last_n_checkpoints: int = 0):
        super().__init__(frequency, self.checkpoint, after_training=True)
        self.__static_name__ = name
        self.permanent_checkpoints: int = permanent_checkpoints

        self.ith_checkpoint: int = 0
        self.max_checkpoints: int = keep_last_n_checkpoints

    def checkpoint(self):
        assert self.trainer.__getattribute__('model') is not None, 'trainer has no model to checkpoint'

        model: BaseModel = self.trainer.model

        model.checkpoint(name=self.name, state=self.build_state())

        # TODO - really we only need to checkpoint once, then copy/paste the file.
        if self.max_checkpoints > 0:
            ith = '' if self.max_checkpoints == 0 else f'_{self.ith_checkpoint + 1}'
            model.checkpoint(name=f'{self.name}{ith}', state=self.build_state())
            self.ith_checkpoint = (self.ith_checkpoint + 1) % self.max_checkpoints

        if self.permanent_checkpoints > 0 and self.trainer.iter > 0 and self.trainer.iter % self.permanent_checkpoints == 0:
            self.trainer.log.info("Permanent checkpoint hit... saving model again.")
            model.checkpoint(name=f'{self.name}@{self.trainer.iter}', state=self.build_state())

    def build_state(self):
        assert self.trainer.__getattribute__('model') is not None, 'trainer does not have the model to checkpoint'

        state = {}

        optimizer: optim.Optimizer = self.trainer.optimizer
        state['optim_state'] = optimizer.state_dict()

        state['iteration'] = self.trainer.iter

        # You can override this if checkpoints need to hold specific data.
        return state

    @property
    def name(self):
        # Overwrite this if you want it to be a name that can change based on iterations or something
        return self.__static_name__

    def on_resume(self):
        assert self.trainer.__getattribute__('model') is not None, 'trainer has no model to checkpoint'

        model: BaseModel = self.trainer.model
        state = model.load(name=self.name)

        if 'optim_state' in state:
            self.trainer.optimizer.load_state_dict(state['optim_state'])

        if 'iteration' in state:
            self.trainer.start_iter = state['iteration']
            self.trainer.iter = state['iteration']

        return state

class TrainingVisualizationHook(PeriodicStepHook):

    def __init__(self, frequency: int = 20, batch: bool = True):
        super().__init__(frequency=frequency)
        self.visualizer = Visualization()
        self.batch = batch

        self.freq_mult = 1
        self.failed = False
        self.since_failure = 0

    @periodic
    def after_iteration(self):
        loss_item: List[float] = self.trainer.storage.find_item("loss", self.frequency)
        if self.batch:
            loss_item = [sum(self.trainer.storage.find_item("loss", self.frequency)) / self.frequency]

        success = self.visualizer.add_data(loss_item)

        # If we did not succeed in graphing, increase the frequency we call this hook by 2
        # Continue to do this 5 times before setting an unreasonably high frequency resulting in it never being called
        # if no failure state is seen for the next 10 calls to the hook, turn the freq mult. down by 2 until its 1 again
        # if the freq mult. reaches 1 again, the failed flag is set to false and this won't run
        if not success:
            if self.freq_mult >= 32: # 2 ^ 5
                self.frequency = 999_999_999 # never run again

            self.frequency *= 2

            self.failed = True
            self.since_failure = 0
            self.freq_mult *= 2
        elif self.failed:
            self.since_failure += 1

            if self.since_failure > 10:
                self.frequency /= 2
                self.freq_mult /= 2

                self.since_failure = 0

                if self.freq_mult == 1:
                    self.failed = False


class LogTrainingLoss(PeriodicStepHook):
    def __init__(self, frequency: int = 20):
        super().__init__(frequency=frequency)

    @periodic
    def after_iteration(self):
        loss_item: float = sum(self.trainer.storage.find_item("loss", self.frequency)) / self.frequency
        data_delta: float = sum(self.trainer.storage.find_item("data_delta", self.frequency)) / self.frequency
        inference_delta: float = sum(self.trainer.storage.find_item("inference_delta", self.frequency)) / self.frequency
        optim_delta: float = sum(self.trainer.storage.find_item("optim_delta", self.frequency)) / self.frequency
        step_delta: float = sum(self.trainer.storage.find_item("step_delta", self.frequency)) / self.frequency
        back_prop_delta: float = sum(self.trainer.storage.find_item("back_prop_delta", self.frequency)) / self.frequency

        loss_items = []
        for key in self.trainer.storage.events:
            if str(key).endswith("_loss"):
                loss_items.append([key, sum(self.trainer.storage.find_item(key, self.frequency)) / self.frequency])

        loss_string = f'loss {loss_item:.4f},'
        for item in loss_items:
            loss_string += f' {item[0]} {item[1]:.4f},'
        training_log.info(f"Step {self.trainer.iter}, {loss_string} data delta {data_delta:.4f}s, inf delta: {inference_delta:.4f}s, back prop delta: {back_prop_delta:.4f}s, optim delta: {optim_delta:.4f}s, delta for step: {step_delta:.4f}s")
