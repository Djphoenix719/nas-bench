import copy
from typing import Dict

from trials.Evolution import mutate_fn
from trials.Search import search
from trials.Selection import drop_worst_fn
from trials.Selection import sel_tournament_fn
from trials.Utilities import random_spec


def main():
    # 1 month of GPU time
    MAX_TIME = 60 * 60 * 24 * 31
    # number of best models to maintain
    N_BEST = 10
    # size of the population that should be maintained
    N_POP = 250
    # number of trials that should be performed with a single set of parameters
    # note trials are deterministic but may differ slightly due to multiple samples
    # present in the the nasbench dataset.
    N_EPOCH = 1

    INITIAL_POPULATION = [random_spec() for idx in range(N_POP)]

    BASE_TRIAL_ARGS = {
        "max_time": MAX_TIME,
        "num_best": N_BEST,
        "num_epochs": N_EPOCH,
    }

    def banner(value: str, min_width: int = 50, character: str = "-") -> None:
        width = max(min_width, len(value) + 2)

        print(character * width)
        print(character + value.center(width - 2) + character)
        print(character * width)

    def extend_trial(**kwargs) -> Dict:
        population = copy.deepcopy(INITIAL_POPULATION)
        return {**BASE_TRIAL_ARGS, "initial_population": population, **kwargs}

    trials = {
        "test": extend_trial(
            **{
                "mut_fn": mutate_fn(1.0),
                "sel_fn": sel_tournament_fn(int(N_POP * 0.25), int(N_POP * 0.1)),
                "drp_fn": drop_worst_fn(int(N_POP * 0.1)),
            }
        )
    }

    for name, params in trials.items():
        banner(name)
        search(**params)


if __name__ == "__main__":
    main()
