from collections import OrderedDict
import csv
from logging import Logger
import pickle
from random import Random
from typing import List, Optional, Set, Tuple, Union
import os

from rdkit import Chem
import numpy as np
from tqdm import tqdm

from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.scaffold import log_scaffold_stats, scaffold_split
from chemprop.features import load_features, load_valid_atom_features


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def preprocess_smiles_columns(path: str,
                              smiles_columns: Optional[Union[str, List[Optional[str]]]],
                              number_of_molecules: int = 1) -> List[Optional[str]]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if not isinstance(smiles_columns, list):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError(
                    'Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError(
                    'Provided smiles_columns do not match the header of data file.')

    return smiles_columns


def get_task_names(path: str,
                   smiles_columns: Union[str, List[str]] = None,
                   target_columns: List[str] = None,
                   ignore_columns: List[str] = None) -> List[str]:
    """
    Gets the task names from a data CSV file.
    If :code:`target_columns` is provided, returns `target_columns`.
    Otherwise, returns all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :return: A list of task names.
    """
    if target_columns is not None:
        return target_columns

    columns = get_header(path)

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(
            path=path, smiles_columns=smiles_columns)

    ignore_columns = set(
        smiles_columns + ([] if ignore_columns is None else ignore_columns))

    target_names = [
        column for column in columns if column not in ignore_columns]

    return target_names


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.
    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :return: A :class:`~chemprop.data.MoleculeDataset` with only the valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol)])


def get_data(path: str,
             smiles_columns: Union[str, List[str]] = None,
             target_columns: List[str] = None,
             ignore_columns: List[str] = None,
             skip_invalid_smiles: bool = True,
             features_path: List[str] = None,
             features_generator: List[str] = None,
             atom_descriptors_path: str = None,
             max_data_size: int = None,
             store_row: bool = False,
             logger: Logger = None,
             skip_none_targets: bool = False) -> MoleculeDataset:
    """
    Gets SMILES and target values from a CSV file.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_column` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`.
    :param args: Arguments, either :class:`~chemprop.args.TrainArgs` or :class:`~chemprop.args.PredictArgs`.
    :param features_path: A list of paths to files containing features. If provided, it is used
                          in place of :code:`args.features_path`.
    :param features_generator: A list of features generators to use. If provided, it is used
                               in place of :code:`args.features_generator`.
    :param atom_descriptors_path: The path to the file containing the custom atom descriptors.
    :param max_data_size: The maximum number of data points to load.
    :param logger: A logger for recording output.
    :param store_row: Whether to store the raw CSV row in each :class:`~chemprop.data.data.MoleculeDatapoint`.
    :param skip_none_targets: Whether to skip targets that are all 'None'. This is mostly relevant when --target_columns
                              are passed in, so only a subset of tasks are examined.
    :return: A :class:`~chemprop.data.MoleculeDataset` containing SMILES and target values along
             with other info such as additional features when desired.
    """
    debug = logger.debug if logger is not None else print

    smiles_columns = smiles_columns
    target_columns = target_columns
    ignore_columns = ignore_columns
    features_path = features_path
    features_generator = features_generator
    atom_descriptors_path = atom_descriptors_path
    max_data_size = max_data_size

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(
            path=path, smiles_columns=smiles_columns)

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            # each is num_data x num_features
            features_data.append(load_features(feat_path))
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    # Load data
    with open(path) as f:
        reader = csv.DictReader(f)

        # By default, the targets columns are all the columns except the SMILES column
        if target_columns is None:
            target_columns = get_task_names(
                path=path,
                smiles_columns=smiles_columns,
                target_columns=target_columns,
                ignore_columns=ignore_columns,
            )

        all_smiles, all_targets, all_rows, all_features = [], [], [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]

            targets = [float(row[column]) if row[column] !=
                       '' else None for column in target_columns]

            # Check whether all targets are None and skip if so
            if skip_none_targets and all(x is None for x in targets):
                continue

            all_smiles.append(smiles)
            all_targets.append(targets)

            if features_data is not None:
                all_features.append(features_data[i])

            if store_row:
                all_rows.append(row)

            if len(all_smiles) >= max_data_size:
                break

        atom_features = None
        atom_descriptors = None
        if atom_descriptors is not None:
            try:
                descriptors = load_valid_atom_features(
                    atom_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(
                    f'Failed to load or valid custom atomic descriptors: {e}')

            if atom_descriptors == 'feature':
                atom_features = descriptors
            elif atom_descriptors == 'descriptor':
                atom_descriptors = descriptors

        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                targets=targets,
                row=all_rows[i] if store_row else None,
                features_generator=features_generator,
                features=all_features[i] if features_data is not None else None,
                atom_features=atom_features[i] if atom_features is not None else None,
                atom_descriptors=atom_descriptors[i] if atom_descriptors is not None else None,
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                               total=len(all_smiles))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(
                f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data
