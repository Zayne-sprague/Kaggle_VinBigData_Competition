from detectron2.data import DatasetCatalog

from src.data_loaders.data_loader import TrainingDataLoader


class DetectronTrainingDataLoader(TrainingDataLoader):

    def __init__(self, name: str, readin_annotation_data=True, readin_meta_data=True):
        super().__init__(readin_annotation_data=readin_annotation_data, readin_meta_data=readin_meta_data)
        self.name: str = name

    def register_records(self):
        assert len(self.records) > 0, "Need to load in the records before you can register them with detectron"
        DatasetCatalog.register(self.name, lambda: self.records)

    def register_metadata(self):
        pass
