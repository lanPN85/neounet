import pydoc
import os
import yaml

from loguru import logger
from typing import List
from omegaconf import OmegaConf


def load_config(paths: List[str]):
    assert len(paths) > 0
    conf = OmegaConf.load(paths[0])

    for path in paths[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(path))

    return conf


def print_config(config):
    print(OmegaConf.to_yaml(config))


def load_class_from_conf(config, *args, **kwargs):
    logger.info(f"Loading class {config.cls}")
    clazz = pydoc.locate(config.cls)

    if clazz is None:
        raise ValueError(f"Cannot find class {config.cls}")

    return clazz(*args, **kwargs, **OmegaConf.to_container(config.kwargs))


def read_mlflow_auth(path="auth/mlflow.yml"):
    """
    Read username and password from an auth file

    :param path: Path to the auth file (YAML), defaults to ".auth"
    :type path: str, optional
    :return: URL with user/pass for MLFlow
    """
    if not os.path.exists(path):
        return None

    with open(path, "rt") as f:
        d: dict = yaml.full_load(f)

    if "s3_endpoint" in d.keys():
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = d["s3_endpoint"]
    if "aws_key" in d.keys():
        os.environ["AWS_ACCESS_KEY_ID"] = d["aws_key"]
    if "aws_secret" in d.keys():
        os.environ["AWS_SECRET_ACCESS_KEY"] = d["aws_secret"]

    if "user" in d.keys() and "password" in d.keys() and "url" in d.keys():
        return f"{d.get('protocol', 'https')}://{d['user']}:{d['password']}@{d['url']}"

    return None
