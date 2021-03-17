import torch

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
import os

from src.utils.paths import DETECTRON_OUTPUT_DIR, DATA, MODELS_DIR
from detectron_modeling.data.detectron_abnormal_dataset import DetectronTrainingAbnormalDataSet


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def main():
    data_set = DetectronTrainingAbnormalDataSet("training_data")
    data_set.load_records()
    data_set.register_records()
    data_set.register_metadata()

    output_path = str(DETECTRON_OUTPUT_DIR / 'abnormal')
    trained_weights = str(DATA / 'vbd_r50fpn3x_512px/model_final.pth')

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.OUTPUT_DIR = output_path
    cfg.MODEL.WEIGHTS = trained_weights

    cfg.DATASETS.TRAIN = ("training_data")

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 200000
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000

    cfg.TEST.EVAL_PERIOD = 5000

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # cfg.MODEL.ROI_BOX_HEAD.NAME = "MyOutputLayer"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    modelout = build_model(cfg)
    modelin = torch.load(f'{MODELS_DIR}/resnet50_test3.pth')

    trainer = MyTrainer(cfg)
    # trainer.resume_or_load(resume=True)
    # trainer.train()







if __name__ == "__main__":
    main()