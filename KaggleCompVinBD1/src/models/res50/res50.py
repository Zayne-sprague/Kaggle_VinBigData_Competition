import torch
from torch import nn
from torchvision.models import resnet50

from src.models.model import BaseModel


class Res50(BaseModel):
    def __init__(self):
        super().__init__(model_name="Res50")

        self.model = nn.Sequential(*(list(resnet50(True).children()))[:-1])

        self.fc = nn.Linear(2048, 2)
        self.lsoft = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)

        # So... this sucks... may have to find a pretrained resnet on grayscale or do it myself which seems unreasonable
        x = x.repeat(1, 3, 1, 1)

        x = self.model(x)

        x = x.view([batch_size, -1])
        x = self.fc(x)
        predictions = self.lsoft(x)

        return predictions



if __name__ == "__main__":
    from src.data_loaders.abnormal_dataloader import TrainingAbnormalDataLoader
    from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask

    model = Res50()

    print(model)

    dataloader = TrainingAbnormalDataLoader()
    dataloader.load_records()

    training_task = AbnormalClassificationTask("abnormal_classification_task_res50", checkpoint_frequency=1, validation_frequency=1)
    training_task.register_training_data(dataloader, train_to_val_split=0.75)
    training_task.begin_or_resume(model)
