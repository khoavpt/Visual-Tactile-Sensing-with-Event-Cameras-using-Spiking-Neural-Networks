# filepath: /c:/Users/LENOVO/OneDrive/uet/lab/spike-neural-networks/Spiking-neural-networks-for-event-cameras/tactile_experiment/src/train.py
import rootutils
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
import logging
import os
from pytorch_lightning.loggers import TensorBoardLogger

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIG_PATH = str(ROOTPATH / "configs")

import src.utils as utils
from src.data.custom_dataset import create_dataloader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    print(f"Config used: {cfg}")
    data_dir = os.path.join(ROOTPATH, cfg.data_module.output_dir)
    
    # Check if the processed data directory is empty
    if not os.listdir(data_dir):
        logger.error(f"The processed data directory {data_dir} is empty. Please process the data first using process_train_data.py.")
        return

    train_dataloader, test_dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
    )

    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    loss_logger = utils.logging.LossLogger(model_name=cfg.model._target_)
    
    tb_logger = TensorBoardLogger(save_dir=os.path.join(ROOTPATH, "outputs"), name="lightning_logs")

    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[loss_logger], logger=tb_logger)

    # Train model
    trainer.fit(model, train_dataloader, test_dataloader)

    # Log model configs and results
    loss_logger.plot_results(save_path=os.path.join(ROOTPATH, "figures", f"{cfg.model._target_}_results.png"))
    logger.info(f"Data config: {cfg.data_module}")
    logger.info(f"Model config: {cfg.model}")
    logger.info(f"Trainer config: {cfg.trainer}")
    
    logger.info(f"Train losses: {loss_logger.train_losses}")
    logger.info(f"Train accuracies: {loss_logger.train_accuracies}")
    logger.info(f"Train F1 scores: {loss_logger.train_f1s}")
    logger.info(f"Validation losses: {loss_logger.val_losses}")
    logger.info(f"Validation accuracies: {loss_logger.val_accuracies}")
    logger.info(f"Validation F1 scores: {loss_logger.val_f1s}")
    logger.info(f"Epoch durations: {loss_logger.epoch_durations}")

    return

if __name__ == "__main__":
    main()