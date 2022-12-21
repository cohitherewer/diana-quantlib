import copy

import pytorch_lightning as pl

from DianaModules.utils.compression.ModelDistiller import QModelDistiller


class DistillObjective:
    def __init__(
        self,
        student,
        teacher,
        datamodule,
        trainer_args,
        seed=42,
        base_params=None,
    ):
        self.trainer_args = trainer_args
        self.data_dir = ""
        self.batch_size = 128
        self.datamodule = datamodule
        self.student = student
        self.teacher = teacher
        self.base_params = base_params or {}
        self.seed = seed

    def __call__(self, trial):
        pl.seed_everything(42)
        params = self.base_params.copy()

        params["momentum"] = (trial.suggest_uniform("momentum", 0.0, 0.99),)
        params["weight_decay"] = (
            trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
        )
        params["learning_rate"] = (
            trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        )
        params["nesterov"] = (
            trial.suggest_categorical("nesterov", ["True", "False"]) == "True"
        )

        scheduler_name = trial.suggest_categorical(
            "scheduler", ["multistep", "cosine"]
        )
        if scheduler_name == "cosine":
            params["lr_scheduler"] = "cosine"

            params["gamma"] = 1.0
            params["warm_up"] = 0

        else:
            params["lr_scheduler"] = "multistep"
            params["gamma"] = trial.suggest_uniform("gamma", 0.1, 0.5)
            params["warm_up"] = trial.suggest_int("warm_up", 0, 2000, 250)

        model = QModelDistiller(
            student=self.student,  # make sure we start from same init
            teacher=self.teacher,
            seed=self.seed,
            #            max_epochs=self.trainer_args.max_epochs,
            **params
        )
        # trainer = pl.Trainer.from_argparse_args(self.trainer_args)
        trainer = pl.Trainer(gpus=[0])

        trainer.fit(model, self.datamodule)
        return trainer.checkpoint_callback.best_model_score
