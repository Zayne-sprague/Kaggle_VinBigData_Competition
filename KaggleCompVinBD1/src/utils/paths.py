from pathlib import Path
import platform


system = platform.system()

__CURR_DIR__ = Path(__file__).resolve().parent

CACHE_DIR = __CURR_DIR__ / "../../../caches"
MODELS_DIR = __CURR_DIR__ / "../../../models"
DETECTRON_OUTPUT_DIR = __CURR_DIR__ / '../../../detectron_outputs'
DATA = __CURR_DIR__ / "../../../data"
SUBMISSIONS_DIR = __CURR_DIR__ / "../../../submissions"

TRAINING_ANNOTATION_DATA = DATA / "vbd_rawdata/train.csv"

TRAIN_META_DATA = DATA / "vbd256/train_meta.csv"
TEST_META_DATA = DATA / "vbd_testmeta/test_meta.csv"

if system == "Windows":

    RAW_VBD_DICOM_DATA = Path("G:").resolve() / "VBD_DATA"
    TRAIN_DICOM_DATA = RAW_VBD_DICOM_DATA / "train"

    CONVERTED_VBD_DICOM_DATA_FOLDER = Path("G:").resolve() / "VBD_DATA" / "curated"

else:
    ROOT = Path("/users/zaynesprague")

    RAW_VBD_DICOM_DATA = ROOT / "Kaggle/VBD_DATA"
    TRAIN_DICOM_DATA = RAW_VBD_DICOM_DATA / "train"

    CONVERTED_VBD_DICOM_DATA_FOLDER = ROOT / "Kaggle/VBD_DATA" / "curated"

