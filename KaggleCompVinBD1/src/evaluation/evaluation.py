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


def main(submission_file_name):
    log.info(f'Starting evaluation for submission file {submission_file_name}')

    dataset = TestingMulticlassDataset()
    records = dataset.load_records()

    dataloader = iter(DataLoader(
        dataset=dataset,
        batch_size=1,
    ))

    model_one = RetinaNet()
    model_one.load("retinanet_backbone_test")
    model_one.eval()

    model_two = RetinaNetEnsemble()
    model_two.load("retinaNetEnsemble_FullTestOne")
    model_two.eval()


    total = len(dataset)


    predictions = []
    for _ in tqdm(range(total), total=total, desc='Creating Predictions'):
        batch = next(dataloader)

        id = batch['image_id']

        w, h = batch['width'], batch['height']

        healthy_or_abnormal = torch.argmax(model_one(batch)['preds'], dim=1).tolist()
        if healthy_or_abnormal[0] == 1:
            predictions.append([id, '14 1.0 0 0 1 1'])
        else:
            preds = model_two(batch)
            pred = preds['preds'][0]

            pred_string = ''
            for p_idx in range(len(pred['boxes'])):
                pred_string += f"{pred['labels'][p_idx].item()} {pred['scores'][p_idx].item()} {int(pred['boxes'][p_idx][0].item() / 256.0 * w)} {int(pred['boxes'][p_idx][1].item() / 256.0 * h)} {int(pred['boxes'][p_idx][2].item() / 256.0 * w)}, {int(pred['boxes'][p_idx][3].item() / 256.0 * h)} "

            predictions.append([
                id, pred_string
            ])



    with open(f'{SUBMISSIONS_DIR}/{submission_file_name}.csv', 'w+') as submission:
        submission_writer = csv.writer(submission, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        submission_writer.writerow(['ID', 'TARGET'])

        for pred in predictions:
           submission_writer.writerow(pred)

    log.info(f"Evaluation completed, submission wrote out {len(predictions)} predictions to {SUBMISSIONS_DIR}/{submission_file_name}.csv")

if __name__:
    main("submission_test_1")
