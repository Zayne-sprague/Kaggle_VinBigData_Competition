from src.ConfigWrapper import ConfigWrapper

import logging
import sys
import coloredlogs
from enum import Enum


handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# root logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


coloredlogs.install(level='INFO', logger=log)

# data loader logging
dl_log = logging.getLogger("DATA_LOADER")
dl_log.setLevel(logging.INFO)
dl_log.addHandler(handler)

coloredlogs.install(level='INFO', logger=dl_log)

# modeling logger
model_log = logging.getLogger("MODELER")
model_log.setLevel(logging.INFO)
model_log.addHandler(handler)

coloredlogs.install(level='INFO', logger=model_log)


training_log = logging.getLogger("TRAINING")
training_log.setLevel(logging.INFO)
training_log.addHandler(handler)

coloredlogs.install(level='INFO', logger=training_log)


hooks_log = logging.getLogger("HOOKS")
hooks_log.setLevel(logging.INFO)
hooks_log.addHandler(handler)

coloredlogs.install(level='INFO', logger=hooks_log)


CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Healthy"
]


class Classifications(Enum):
    AorticEnlargement = 0
    Atelectasis = 1
    Calcification = 2
    Cardiomegaly = 3
    Consolidation = 4
    ILD = 5
    Infiltration = 6
    LungOpacity = 7
    NoduleMass = 8
    OtherLesion = 9
    PleuralEffusion = 10
    PleuralThickening = 11
    Pneumothorax = 12
    PulmonaryFibrosis = 13
    Healthy = 14


def is_record_healthy(record):
    return len(record['annotations']) == 0 or all([x['category_id'] == Classifications.Healthy.value for x in record['annotations']])


config = ConfigWrapper()
