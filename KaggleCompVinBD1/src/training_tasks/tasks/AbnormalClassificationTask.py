import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from tabulate import tabulate

from src import config
from src.models.model import BaseModel
from src.data_loaders.abnormal_dataloader import TrainingAbnormalDataLoader
from src.training_tasks.training_task import SimpleTrainer


class AbnormalClassificationTask(SimpleTrainer):

    def validation(self, dataloader: TrainingAbnormalDataLoader, model: BaseModel) -> dict:
        model.eval()

        self.log.info("Beginning Validation")

        dataloader.display_metrics(dataloader.get_metrics())

        data = iter(DataLoader(dataloader, batch_size=config.batch_size, num_workers=4))
        total = len(dataloader) // config.batch_size + 1

        # idx 0 == correct, idx 1 == incorrect
        stats = {
            'healthy': [0, 0],
            'abnormal': [0, 0]
        }

        labels = ['healthy', 'abnormal']

        for _, i in tqdm(enumerate(range(total)), total=len(range(total)), desc="Validating the model"):
            batch = next(data)

            for ky, val in batch.items():
                # If we can, try to load up the batched data into the device (try to only send what is needed)
                if isinstance(batch[ky], torch.Tensor):
                    batch[ky] = batch[ky].to(config.devices[0])

            y: torch.Tensor = batch['label']

            predictions = torch.argmax(model(batch)['preds'], 1)

            for idx, prediction in enumerate(predictions.tolist()):
                if prediction == y[idx]:
                    stats[labels[y[idx]]][0] += 1
                else:
                    stats[labels[y[idx]]][1] += 1

        table = []
        for stat in stats:
            table.append([stat, stats[stat][0], stats[stat][1]])

        self.log.info(f'\n-- Validation Report --\n{tabulate(table, headers=["Type","Correct","Incorrect"])}')

        model.train()

        return stats

