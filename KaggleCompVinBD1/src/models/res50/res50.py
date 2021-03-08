import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from src.models.model import BaseModel
from src.models.components.ResNet import IdentityBlock


class Res50(BaseModel):
    def __init__(self):
        super().__init__(model_name="Res50")

        self.C1 = nn.Conv2d(1, 64, (1,1), (1,1))
        self.block = IdentityBlock(64, 64, 256)
        self.FC1 = nn.Linear(4209, 1000)
        self.FC2 = nn.Linear(1000, 2)

    def forward(self, x):
        batch_size = int(x.shape[0])
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)

        x = self.C1(x)
        x = self.block(x)

        x = x.view([batch_size, -1])

        x = self.FC1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.FC2(x)

        predictions = self.LSoftmax(x)
        return predictions



if __name__ == "__main__":
    from src.data_loaders.abnormal_dataloader import TrainingAbnormalDataLoader
    from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask

    model = Res50()
    dataloader = TrainingAbnormalDataLoader()
    dataloader.load_records()

    training_task = AbnormalClassificationTask("abnormal_classification_task_res50", checkpoint_frequency=1, validation_frequency=1)
    training_task.register_training_data(dataloader, train_to_val_split=0.75)
    training_task.begin_or_resume(model)
