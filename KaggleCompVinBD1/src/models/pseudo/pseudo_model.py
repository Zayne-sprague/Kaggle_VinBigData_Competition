import torch

from src.models.model import BaseModel


class PseudoModel(BaseModel):

    def __init__(self, num_classes: int= 2, model_name: str = "SimpleNet"):
        super().__init__(model_name=model_name)

        self.num_classes: int = num_classes

        self.fc = torch.nn.Linear(2, self.num_classes)

    def forward(self, x: torch.Tensor):

        return torch.rand([x.shape[0], self.num_classes])


if __name__ == "__main__":
    from src.data_loaders.pseudo_dataloader import TrainingPseudoDataloader
    from src.training_tasks.tasks.PseudoTask import PseduoTask

    model = PseudoModel()
    dataloader = TrainingPseudoDataloader()
    dataloader.load_records()

    training_task = PseduoTask("pseudo_classification_task")
    training_task.register_training_data(dataloader, train_to_val_split=0.75)
    training_task.begin_or_resume(model)
