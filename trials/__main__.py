import copy
import math
from typing import Set
from typing import Callable

from trials.Constants import *
from trials.Evolution import *
from trials.ModelSpec import *
from trials.Search import *
from trials.Selection import *
from trials.Utilities import *

def main():
    def search(
            max_time: float,
            num_best: int,
            num_epochs: int,
            initial_population: List[SpecWrapper],
            mut_fn: Callable[[List[SpecWrapper]], List[SpecWrapper]],
            sel_fn: Callable[[List[SpecWrapper]], List[SpecWrapper]],
            drp_fn: Callable[[List[SpecWrapper]], List[SpecWrapper]]
    ):
        sel_best = sel_best_fn(num_best)

        def run_epoch(epoch_num: int, population: List[SpecWrapper]):
            reset_trial_stats(RNG_SEED + epoch_num)

            # initialize our population list, update budget counters
            # this also effectively "trains" the initial population
            population = [get_spec(ind.get_hash()) for ind in population]

            # desired size of the population
            p_size = len(population)

            # list of hashes that we have previously evaluated which can be skipped in the future
            done: Set[str] = set()

            # running cumulative time total of all epochs
            cur_time: float = 0

            def print_update():
                nonlocal cur_time, max_time, done
                print(f"{cur_time / 1000:0.2f}k/{max_time / 1000:0.0f}k ({(cur_time / max_time) * 100:0.2f}%) seconds simulated, {len(done)} unique models")

            # update done, adding any new hashes to our set
            def update_done(items: List[SpecWrapper]):
                nonlocal cur_time
                done.update(map(lambda x: x.get_hash(), items))
                cur_time = sum([ind.get_data().train_time for ind in map(lambda x: get_spec(x), done)])

            update_done(population)
            print_update()

            while cur_time < max_time:
                # drop some candidates base on the specified function
                population = drp_fn(population)
                assert len(population) > 0
                update_done(population)

                # number of new specs to generate through mutation or crossover
                num_new = p_size - len(population)
                new_specs: List[SpecWrapper] = []
                while len(new_specs) < num_new:
                    # select some candidates to mutate
                    candidates = sel_fn(population)
                    # mutate the candidates with the fn
                    candidates = mut_fn(candidates)

                    # only "evaluate" candidates who were not evaluate
                    # so if we hit a duplicate candidate, we should not
                    # add more training time, as it is wasteful
                    cand_hashes = [cand.get_hash() for cand in candidates]
                    cand_hashes = [hsh for hsh in cand_hashes if hsh not in done]

                    # remove dupe hashes
                    candidates = [get_spec(hsh) for hsh in cand_hashes]

                    # update the list of new specs
                    new_specs = [*new_specs, *candidates]

                population = [*population, *new_specs][:p_size]

                # [ind.get_data() for ind in population]
                update_done(population)

                print_update()

            best = sel_best(population)
            best.sort()

            abs_best = best[-1].get_data()
            print(f'Best in trial -- Test: {abs_best.test_accuracy:0.7f}, Valid: {abs_best.valid_accuracy:0.7f}')

            return population, best, done

        results: [List[List[SpecWrapper], List[SpecWrapper], Set[str]]] = []
        for epoch_num in range(num_epochs):
            print('-' * 50)
            population, best, done = run_epoch(epoch_num, initial_population)
            results.append([population, best, done])
            print(f'Finished evaluation, {len(done)} models evaluated')
            print('-' * 50)

        return results

    # 365 days of GPU time
    MAX_TIME = 60 * 60 * 24 * 31
    N_BEST = 10
    N_POP = 250
    N_EPOCH = 1

    INITIAL_POPULATION = [random_spec() for idx in range(N_POP)]

    trials = {
        'test': {
            "max_time": MAX_TIME,
            "num_best": N_BEST,
            "num_epochs": N_EPOCH,
            "initial_population": copy.deepcopy(INITIAL_POPULATION),
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_tournament_fn(int(N_POP * 0.25), int(N_POP * 0.1)),
            "drp_fn": drop_worst_fn(int(N_POP * 0.1))
        }
    }

    for key, values in trials.items():
        print(key)
        search(**values)


if __name__ == '__main__':
    main()