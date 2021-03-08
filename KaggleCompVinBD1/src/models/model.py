import torch
from torch.nn import Module

from typing import Optional
import uuid
import logging

from src.utils.paths import MODELS_DIR
from src import model_log


class BaseModel(Module):

    def __init__(self,
                 model_name: str,
                 ):
        super().__init__()

        self.log: logging.Logger = model_log

        self.model_name = model_name or f'{uuid.uuid4()}'
        self.log.info(f"Creating model named: {self.model_name}")


    def checkpoint(self, name: Optional[str] = None, state: Optional[dict] = None) -> None:
        self.log.info("Checkpoint hit, saving model")

        checkpoint_path = f"{MODELS_DIR}/{name or self.model_name}.pth"

        if not state:
            state = {}

        try:
            torch.save({
                **state,
                'model_state_dict': self.state_dict(),
            }, checkpoint_path)

            self.log.info(f"Checkpoint saved to {checkpoint_path}")

        except Exception as e:
            self.log.error(f"Failure to save model to pth file {checkpoint_path}")
            self.log.error(e)

    def load(self, name: Optional[str] = None) -> dict:
        name = name or self.model_name
        checkpoint_path = f'{MODELS_DIR}/{name}.pth'

        self.log.info(f"Loading checkpoint for model {name}")

        try:
            state = torch.load(checkpoint_path)
            self.load_state_dict(state['model_state_dict'])

            return state
        except Exception as e:
            self.log.error(f"Failure to load model from pth file {checkpoint_path}")
            self.log.error(e)

            return {}

