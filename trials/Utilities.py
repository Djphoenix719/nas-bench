import os
import random

import numpy as np
import pandas as pd

from typing import List
from typing import Union

from trials.Constants import *
from trials.ModelSpec import SpecWrapper


def make_specs(population: List[str]) -> List[SpecWrapper]:
    """
    Transform a list of hashes into a population of SpecWrappers.
    :param population: The population to transform
    :return: A list of SpecWrappers.
    """

    return [SpecWrapper(ind) for ind in population]


def get_spec(spec_hash: str) -> Union[SpecWrapper, None]:
    """
    Fetch a spec based on the hash.
    :param spec_hash: The hash of the spec.
    :return: A SpecWrapper of the spec or None if it doesn't exist.
    """
    if not spec_hash in all_specs:
        return None

    spec = nasbench.get_metrics_from_hash(spec_hash)
    matrix: np.ndarray = spec[0]["module_adjacency"]
    ops: List[str] = spec[0]["module_operations"]
    return SpecWrapper(matrix=matrix, ops=ops)


def random_spec() -> SpecWrapper:
    """
    Return a random spec.
    :return: A SpecWrapper for the spec.
    """
    return get_spec(random.choice(all_specs))


def build_experiment_results(name: str, population: List[SpecWrapper], best: List[SpecWrapper], all: List[SpecWrapper]):
    """
    Return DataFrames with extraneous columns removed containing data for specs in the population, best, and all sets.
    :param population:
    :param best:
    :param all:
    :return: DataFrames for each set.
    """

    def construct(items: List[SpecWrapper]) -> [pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(list(map(lambda x: x.get_data(), items)))
        df = df.drop(columns=['matrix', 'operations', 'hash'])
        df['total_accuracy'] = df.apply(lambda row: (row['test_accuracy'] + row['valid_accuracy']) / 2, axis=1)
        df = df.sort_values(by=["total_accuracy"], ascending=False)
        stats = gen_stats_table(df)
        return df, stats

    pdf, pdf_stats = construct(population)
    bdf, bdf_stats = construct(best)
    adf, adf_stats = construct(all)

    write_table_html(pdf, f'{name}-PopSpecs.html')
    write_table_html(bdf, f'{name}-BestSpecs.html')
    write_table_html(adf, f'{name}-AllSpecs.html')
    write_table_html(pdf_stats, f'{name}-PopStats.html')
    write_table_html(bdf_stats, f'{name}-BestStats.html')
    write_table_html(adf_stats, f'{name}-AllStats.html')

    return pdf, bdf, adf


def write_table_html(df: pd.DataFrame, name: str, over_write: bool = False) -> None:
    """
    Write a table of html, optionally over-writing existing tables.
    :param df: The dataframe to write to file.
    :param name: The output file name.
    :param over_write: Over-write the file if it already exists.
    """

    path = os.path.join(OUTPUT_FOLDER, name)
    if os.path.exists(path) and not over_write:
        return

    with open(path, 'w') as handle:
        handle.write(df.to_html())


def gen_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a new DataFrame from an existing containing statistical columns with proper names.
    :param df: The source data.
    :return: Resultant DataFrame.
    """
    stats = pd.DataFrame({
        'Min': df.min(),
        'Max': df.max(),
        'Median': df.median(),
        'Mean': df.mean(),
        'Std': df.std(),
    })
    stats = stats.rename(index={
        'parameters': "# Param.",
        'train_time': "Time",
        'train_accuracy': "Trn. Acc.",
        'valid_accuracy': "Val. Acc.",
        'test_accuracy': "Tst. Acc.",
        'total_accuracy': "Ttl. Acc."
    })
    return stats


def reset_trial_stats(seed: int):
    """
    Resets RNG and budget counters to allow for deterministic & reproducible trials.
    """

    random.seed(seed)
    np.random.seed(seed)
    nasbench.reset_budget_counters()

