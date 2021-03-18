import torch

from src.modeling.losses.NLLLossOHE import NLLLossOHE
from src.modeling.models.model import BaseModel


class PseudoModel(BaseModel):

    def __init__(self, num_classes: int= 2, model_name: str = "SimpleNet"):
        super().__init__(model_name=model_name)

        self.num_classes: int = num_classes

        self.fc = torch.nn.Linear(1, self.num_classes)

        self.criterion = NLLLossOHE()

    def forward(self, data: dict):

        state = {'preds': self.fc(torch.unsqueeze(data['image'], 1).float())}
        losses = self.loss(state, data)
        state['losses'] = losses
        return state

    def loss(self, predictions: dict, data: dict) -> dict:
        return {'loss': self.criterion(predictions['preds'], data['label'])}


if __name__ == "__main__":
    from src.data.pseudo_dataset import TrainingPseudoDataSet
    from src.training_tasks.tasks.PseudoTask import PseduoTask

    model = PseudoModel()
    dataloader = TrainingPseudoDataSet()
    dataloader.load_records()

    training_task = PseduoTask("pseudo_classification_task")
    training_task.register_training_data(dataloader, train_to_val_split=0.75)
    training_task.begin_or_resume(model)
