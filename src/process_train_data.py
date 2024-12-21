import rootutils
import hydra
from omegaconf import DictConfig
import logging
import os

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIG_PATH = str(ROOTPATH / "configs")

import src.utils as utils
from src.data.preprocessing import aedat4_to_sequences

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    print(f"Config used: {cfg.data_module}")
    raw_data_dir = os.path.join(ROOTPATH, cfg.data_module.input_dir)
    seq_data_dir = os.path.join(ROOTPATH, cfg.data_module.output_dir)

    logger.info(f"Processing raw .aedat4 data from {raw_data_dir} and saving sequences to {seq_data_dir}")
    aedat4_to_sequences(
        input_dir=raw_data_dir,
        output_dir=seq_data_dir,
        duration=cfg.data_module.frame_duration,
        sequence_length=cfg.data_module.sequence_length,
        encoding_type=cfg.data_module.encoding_type,
    )

if __name__ == "__main__":
    main()