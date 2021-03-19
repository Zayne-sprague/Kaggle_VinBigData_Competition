import torch
from torch.utils.data import DataLoader

from map_boxes import mean_average_precision_for_boxes
from tqdm import tqdm
from tabulate import tabulate

from src import config, Classifications
from src.modeling.models.model import BaseModel
from src.data.abnormal_dataset import TrainingAbnormalDataSet
from src.training_tasks.training_task import SimpleTrainer


class MulticlassDetectionTask(SimpleTrainer):

    def validation(self, dataloader: TrainingAbnormalDataSet, _model: BaseModel) -> dict:
        from src.modeling.models.retinaNetEnsemble.retinaNetEnsemble import RetinaNetEnsemble

        model = RetinaNetEnsemble()
        model.load_state_dict(_model.state_dict())
        model.to(config.validation_device)
        model.eval()

        self.log.info("Beginning Validation")

        dataloader.display_metrics(dataloader.get_metrics())

        data = iter(DataLoader(dataloader, batch_size=config.batch_size, num_workers=4, collate_fn=self.collater))
        total = (len(dataloader) // config.batch_size) + 1

        # idx 0 == correct, idx 1 == incorrect
        stats = {
            'healthy': [0, 0],
            'abnormal': [0, 0]
        }

        classes = list(Classifications)
        labels = [x.name for x in classes]

        det = []
        ann = []

        image_id = 0
        image_id = 0

        for _, i in tqdm(enumerate(range(total)), total=len(range(total)), desc="Validating the model"):
            batch = next(data)

            for ky, val in batch.items():
                # If we can, try to load up the batched data into the device (try to only send what is needed)
                if isinstance(batch[ky], torch.Tensor):
                    batch[ky] = batch[ky].to(config.validation_device)

            predictions = model(batch)['preds']

            for idx, pred in enumerate(predictions):
                annotation = batch['annotations'][idx]

                for p_idx in range(len(pred['boxes'])):
                    det.append([f'{image_id}', pred['labels'][p_idx].item(), pred['scores'][p_idx].item(), pred['boxes'][p_idx][0].item() / 256.0, pred['boxes'][p_idx][1].item() / 256.0, pred['boxes'][p_idx][2].item() / 256.0, pred['boxes'][p_idx][3].item() / 256.0])

                for a_idx in range(len(batch['annotations'][idx]['boxes'])):
                    ann.append([f'{image_id}', torch.argmax(annotation['labels'][a_idx], 0).item(), annotation['boxes'][a_idx][0].item() / 256.0, annotation['boxes'][a_idx][1].item() / 256.0, annotation['boxes'][a_idx][2].item() / 256.0, annotation['boxes'][a_idx][3].item() / 256.0])

                image_id += 1

        for idx in range(len(ann)):
            ann[idx][1] = labels[ann[idx][1]]
        for idx in range(len(det)):
            det[idx][1] = labels[det[idx][1]]

        mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det)


        # table = []
        # for stat in stats:
        #     table.append([stat, stats[stat][0], stats[stat][1]])
        #
        # self.log.info(f'\n-- Validation Report --\n{tabulate(table, headers=["Type", "Correct", "Incorrect"])}')

        return stats
