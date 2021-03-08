import torch
from torch import nn
import torch.nn.functional as F

from src.models.model import BaseModel


class SimpleNet(BaseModel):

    def __init__(self, model_name: str = "SimpleNet"):
        super().__init__(model_name=model_name)

        self.C1 = nn.Conv2d(1, 16, (3, 3), stride=(1, 1))
        self.C2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1))
        self.C3 = nn.Conv2d(32, 32, (3, 3), stride=(1,1))

        self.FC1 = nn.Linear(28_800, 4_000)
        self.FC2 = nn.Linear(4_000, 2)

        self.LSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor):
        batch_size = int(x.shape[0])
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)
        p1 = self.C1(x)
        p1 = F.max_pool2d(p1, 2)

        p2 = self.C2(p1)
        p2 = F.max_pool2d(p2, 2)

        p3 = self.C3(p2)
        p3 = F.max_pool2d(p3, 2)

        p3 = F.relu(p3)

        p3 = p3.view([batch_size, -1])

        x = self.FC1(p3)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.FC2(x)

        predictions = self.LSoftmax(x)
        return predictions


if __name__ == "__main__":
    from src.data_loaders.abnormal_dataloader import TrainingAbnormalDataLoader
    from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask

    model = SimpleNet()
    dataloader = TrainingAbnormalDataLoader()
    dataloader.load_records()

    training_task = AbnormalClassificationTask("abnormal_classification_task", checkpoint_frequency=1, validation_frequency=1)
    training_task.register_training_data(dataloader, train_to_val_split=0.75)
    training_task.begin_or_resume(model)
