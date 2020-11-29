"""
Adopted from AllenNLP:
    https://github.com/allenai/allennlp/blob/v0.7.0/allennlp/models/archival.py
Helper functions for archiving models and restoring archived models.
"""

from typing import NamedTuple, Dict, Any
import json
import logging
import os
import tempfile
import tarfile
import shutil

from stog.utils.file import cached_path
from stog.utils.params import Params
from stog.models.model import Model, _DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# An archive comprises a Model and its experimental config
Archive = NamedTuple("Archive", [("model", Model), ("config", Params)])

# We archive a model by creating a tar.gz file with its weights, config, and vocabulary.
#
# We also may include other arbitrary files in the archive. In this case we store
# the mapping { flattened_path -> filename } in ``files_to_archive.json`` and the files
# themselves under the path ``fta/`` .
#
# These constants are the *known names* under which we archive them.
CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"
_FTA_NAME = "files_to_archive.json"


def archive_model(serialization_dir: str, weights: str = _DEFAULT_WEIGHTS):
    """
    Archive the model weights, its training configuration, and its
    vocabulary to `model.tar.gz`. Include the additional ``files_to_archive``
    if provided.
    Parameters
    ----------
    serialization_dir: ``str``
        The directory where the weights and vocabulary are written out.
    weights: ``str``, optional (default=_DEFAULT_WEIGHTS)
        Which weights file to include in the archive. The default is ``best.th``.
    """
    weights_file = os.path.join(serialization_dir, weights)
    if not os.path.exists(weights_file):
        logger.error("weights file %s does not exist, unable to archive model", weights_file)
        return

    config_file = os.path.join(serialization_dir, CONFIG_NAME)
    if not os.path.exists(config_file):
        logger.error("config file %s does not exist, unable to archive model", config_file)

    archive_file = os.path.join(serialization_dir, "model.tar.gz")
    logger.info("archiving weights and vocabulary to %s", archive_file)
    with tarfile.open(archive_file, 'w:gz') as archive:
        archive.add(config_file, arcname=CONFIG_NAME)
        archive.add(weights_file, arcname=_WEIGHTS_NAME)
        archive.add(os.path.join(serialization_dir, "vocabulary"),
                    arcname="vocabulary")


def load_archive(archive_file: str,
                 device=None,
                 weights_file: str = None) -> Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.
    Parameters
    ----------
    archive_file: ``str``
        The archive file to load the model from.
    weights_file: ``str``, optional (default = None)
        The weights file to use.  If unspecified, weights.th in the archive_file will be used.
    device: ``None`` or PyTorch device object.
    """
    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(f"loading archive file {archive_file} from cache at {resolved_archive_file}")

    tempdir = None
    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, 'r:gz') as archive:
            archive.extractall(tempdir)

        serialization_dir = tempdir

    # Load config
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME))
    config.loading_from_archive = True

    if weights_file:
        weights_path = weights_file
    else:
        weights_path = os.path.join(serialization_dir, _WEIGHTS_NAME)

    # Instantiate model. Use a duplicate of the config, as it will get consumed.
    model = Model.load(config,
                       weights_file=weights_path,
                       serialization_dir=serialization_dir,
                       device=device)

    if tempdir:
        # Clean up temp dir
        shutil.rmtree(tempdir)

    return Archive(model=model, config=config)
