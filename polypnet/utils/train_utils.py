import tempfile

from omegaconf import OmegaConf

from .misc import get_current_git_commit


def log_exp_hyperparams(_logger, config):
    """
    Logs the experiment's hyperparameters to MLflow

    :param _logger: The MLflowLogger to use
    :param config: The Config object
    """
    # Save config file
    with tempfile.NamedTemporaryFile("w+t", suffix=".yml") as f:
        OmegaConf.save(config, f)
        _logger.experiment.log_artifact(
            _logger.run_id, f.name
        )

    model_params = dict([
        (f"model.{k}", v)
        for (k, v) in config.model.kwargs.items()
    ])
    data_params = dict([
        (f"data.{k}", v)
        for (k, v) in config.dataset.kwargs.items()
    ])

    test_data_params = {}
    for item in config.test_datasets:
        test_data_params[f"test_data.{item.name}.class"] = item.cls
        for k, v in item.kwargs.items():
            test_data_params[f"test_data.{item.name}.{k}"] = v

    optimizer_params = dict([
        (f"optimizer.{k}", v)
        for (k, v) in config.optimizer.kwargs.items()
    ])
    scheduler_params = dict([
        (f"scheduler.{k}", v)
        for (k, v) in config.scheduler.kwargs.items()
    ])
    augment_params = dict([
        (f"augment.{k}", v)
        for (k, v) in config.augment.kwargs.items()
    ])

    agc_params = dict([
        (f"agc.{k}", v)
        for (k, v) in config.agc.kwargs.items()
    ])

    # Add loss parameters
    loss_params = dict([
        (f"loss.{k}", v)
        for (k, v) in config.loss.kwargs.items()
    ])
    for i, c in enumerate(config.loss.losses):
        loss_params[f"loss.{i + 1}.class"] = c.cls
        loss_params.update(dict([
            (f"loss.{i + 1}.{k}", v)
            for (k, v) in c.kwargs.items()
        ]))

    label_loss_params = dict([
        (f"label_loss.{k}", v)
        for (k, v) in config.label_loss.kwargs.items()
    ])
    for i, c in enumerate(config.label_loss.losses):
        label_loss_params[f"label_loss.{i + 1}.class"] = c.cls
        label_loss_params.update(dict([
            (f"label_loss.{i + 1}.{k}", v)
            for (k, v) in c.kwargs.items()
        ]))

    wrapper_params = dict([
        (f"wrapper.{k}", v)
        for (k, v) in config.wrapper.kwargs.items()
    ])

    callback_params = {}
    for name, d in config.callbacks.items():
        for k, v in d.items():
            callback_params[f"callback.{name}.{k}"] = v

    _logger.log_hyperparams({
        "val_on_test": config.val_on_test,
        "split_seed": config.split_seed,
        "val_ratio": config.val_ratio,
        "from_segment_model": config.from_segment_model,
        "from_checkpoint": config.trainer.resume_from_checkpoint,
        "data.batch_size": config.loader.batch_size,
        "test_data.batch_size": config.test_loader.batch_size,
        "model.class": config.model.cls,
        "optimizer.class": config.optimizer.cls,
        "scheduler.class": config.scheduler.cls,
        "agc.enabled": config.agc.enabled,
        "git.commit": get_current_git_commit(),
        **model_params,
        **data_params,
        **test_data_params,
        **optimizer_params,
        **augment_params,
        **loss_params,
        **agc_params,
        **scheduler_params,
        **label_loss_params,
        **wrapper_params,
        **callback_params
    })
