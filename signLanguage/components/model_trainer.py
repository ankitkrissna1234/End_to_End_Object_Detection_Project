from ultralytics import YOLO
import sys, os
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.entity.config_entity import ModelTrainerConfig
from signLanguage.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            os.system("unzip Sign_Language_Dataset.zip")
            os.system("rm Sign_Language_Dataset.zip")
            # Load a pretrained model
            model = YOLO("yolo11n.pt")

            # Train the model on your custom dataset
            results = model.train(
                data="data.yaml",
                epochs=self.model_trainer_config.no_epochs,
                imgsz=self.model_trainer_config.image_size,
            )

            os.system("cp runs/detect/train/weights/best.pt runs/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(
                f"cp runs/detect/train/weights/best.pt {self.model_trainer_config.model_trainer_dir}/"
            )

            os.system("rm -rf train")
            os.system("rm -rf test")
            os.system("rm -rf valid")
            os.system("rm -rf data.yaml")
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="runs/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise SignException(e, sys)
