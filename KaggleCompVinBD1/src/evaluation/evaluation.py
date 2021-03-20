import torch
from torch.utils.data import DataLoader

import os
import csv
from tqdm import tqdm

from src.data.multiclass_dataset import TestingMulticlassDataset

from src.modeling.models.retinaNetEnsemble.retinaNetEnsemble import RetinaNetEnsemble
from src.modeling.models.retinaNet.retinaNet import RetinaNet

from src.utils.paths import SUBMISSIONS_DIR
from src import config, log


def main(submission_file_name, model_one_name, model_two_name):
    log.info(f'Starting evaluation for submission file {submission_file_name}')

    dataset = TestingMulticlassDataset()
    dataset.load_records()

    dataloader = iter(DataLoader(
        dataset=dataset,
        batch_size=1,
    ))

    model_one = RetinaNet()
    model_one.to(config.devices[0])
    model_one.load(model_one_name)
    model_one.eval()

    model_two = RetinaNetEnsemble()
    model_two.to(config.devices[0])
    model_two.load(model_two_name)
    model_two.eval()


    total = len(dataset)


    predictions = []
    for i in tqdm(range(total), total=total, desc='Creating Predictions'):


        batch = next(dataloader)
        for ky, val in batch.items():
            # If we can, try to load up the batched data into the device (try to only send what is needed)
            if isinstance(batch[ky], torch.Tensor):
                batch[ky] = batch[ky].to(config.devices[0])

        id = batch['image_id'][0]

        w, h = batch['orig_w'][0].item(), batch['orig_h'][0].item()

        healthy_or_abnormal = torch.argmax(model_one(batch)['preds'], dim=1).tolist()
        if healthy_or_abnormal[0] == 1:
            predictions.append([id, '14 1.0 0 0 1 1'])
        else:
            preds = model_two(batch)
            pred = preds['preds'][0]

            pred_string = ''
            for p_idx in range(len(pred['boxes'])):
                pred_string += f'{pred["labels"][p_idx].item()} {pred["scores"][p_idx].item()} {int(pred["boxes"][p_idx][0].item() / 256.0 * w)} {int(pred["boxes"][p_idx][1].item() / 256.0 * h)} {int(pred["boxes"][p_idx][2].item() / 256.0 * w)} {int(pred["boxes"][p_idx][3].item() / 256.0 * h)} '

            predictions.append([
                id, pred_string
            ])

    with open(f'{SUBMISSIONS_DIR}/{submission_file_name}.csv', 'w+') as submission:
        submission_writer = csv.writer(submission, delimiter=',')

        submission_writer.writerow(['ID', 'TARGET'])
        submission_writer.writerows(predictions)

    log.info(f"Evaluation completed, submission wrote out {len(predictions)} predictions to {SUBMISSIONS_DIR}/{submission_file_name}.csv")

if __name__:
    main("submission_test_1", 'retinaFpnBackbone_realTestone@15000', 'retinaNetEnsemble_FullTestTwo')
