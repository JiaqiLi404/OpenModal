import json
import warnings

import yaml
import pickle
from pathlib import Path
from typing import Union

ALLOWED_FORMATS = ['json', 'yaml', 'pkl']


def load(file,
         file_format=None):
    """Load dataset from json/yaml/pickle files.

    This method provides a unified api for loading dataset from serialized files.

    ``load`` supports loading dataset from serialized files those can be storaged
    in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet
        >>> load('s3://path/of/your/file')  # file is storaged in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split('.')[-1]
    if file_format not in ALLOWED_FORMATS:
        raise TypeError(f'Unsupported format: {file_format}')

    backends = {
        'json': json,
        'yaml': yaml,
        'pkl': pickle
    }
    backend = backends[file_format]

    backend_args={}
    if file_format == 'yaml':
        backend_args['Loader'] = yaml.FullLoader

    if isinstance(file, str):
        with open(file, 'r') as f:
            return backend.load(f, **backend_args)
    elif hasattr(file, 'write'):
        return backend.load(file, **backend_args)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')


def dump(obj,
         file: Union[Path, str, object ] = None,
         file_format=None):
    """Dump dataset to json/yaml/pickle strings or files.

    This method provides a unified api for dumping dataset as strings or to files,
    and also supports custom arguments for each file format.

    ``dump`` supports dumping dataset as strings or to files.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split('.')[-1]
    if file_format not in ALLOWED_FORMATS:
        raise TypeError(f'Unsupported format: {file_format}')

    backends = {
        'json': json,
        'yaml': yaml,
        'pkl': pickle
    }
    backend = backends[file_format]

    if isinstance(file, str):
        with open(file, 'w') as file:
            return backend.dump(obj, file)
    elif hasattr(file, 'write'):
        return backend.dump(obj, file)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
