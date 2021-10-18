import random
from typing import Callable
from typing import List
from trials.ModelSpec import SpecWrapper


def sel_best_fn(n_best: int) -> Callable[[List[SpecWrapper]], List[SpecWrapper]]:
    """
    Select the top k candidates amongst the entire population.
    :param population: The population to select from.
    :param n_best: The number of top individuals to select.
    :return: The selected individuals.
    """

    def fn(population: List[SpecWrapper]):
        # clone the list so we don't sort in place
        population = list(population)
        population.sort()
        return population[-n_best:]

    return fn


def sel_random_fn(n_rand: int) -> Callable[[List[SpecWrapper]], List[SpecWrapper]]:
    """
    Select k individuals from the population at random.
    :param population: The population to select from.
    :param n_rand: The number of individuals to select.
    :return: The selected individuals.
    """

    def fn(population: List[SpecWrapper]):
        return random.sample(population, n_rand)

    return fn


def sel_tournament_fn(
    n_rand: int, n_best: int
) -> Callable[[List[SpecWrapper]], List[SpecWrapper]]:
    """
    Select the n best individuals from amongst k randomly selected candidates.
    :param n_rand: The number of candidates to randomly select.
    :param n_best: The number of top candidates to be returned.
    :return: The selected individuals.
    """

    sel_best = sel_best_fn(n_best)
    sel_rand = sel_random_fn(n_rand)

    def fn(population: List[SpecWrapper]):
        candidates = sel_rand(population)
        return sel_best(candidates)

    return fn


def sel_middle_fn(n_sel: int) -> Callable[[List[SpecWrapper]], List[SpecWrapper]]:
    """
    Select the n middle individuals from amongst the population.
    :param n_sel: The number of candidates to randomly select.
    """
    n_sel = int(n_sel)

    def fn(population: List[SpecWrapper]):
        population = list(population)
        m_idx = int((len(population) - n_sel) / 2)
        return population[m_idx:-m_idx]

    return fn


def drop_worst_fn(n_worst: int) -> Callable[[List[SpecWrapper]], List[SpecWrapper]]:
    """
    Drop the k worst candidates and return the remaining population.
    :param n_worst: Number of worst candidates to drop.
    """
    n_worst = int(n_worst)

    def fn(population: List[SpecWrapper]):
        population = list(population)
        population.sort()
        return population[n_worst:]

    return fn


def drop_oldest_fn(n_oldest: int) -> Callable[[List[SpecWrapper]], List[SpecWrapper]]:
    """
    Drop the k oldest candidates and return the remaining population.
    :param n_oldest: Number of oldest candidates to drop.
    """
    n_oldest = int(n_oldest)

    def fn(population: List[SpecWrapper]):
        population = list(population)
        population.sort(key=lambda x: x.id)
        return population[n_oldest:]

    return fn
