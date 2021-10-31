import copy
import os.path
import random
from typing import Dict
import json

from trials.Evolution import mutate_fn
from trials.Search import search
from trials.Selection import drop_oldest_fn
from trials.Selection import drop_random_fn
from trials.Selection import drop_worst_fn
from trials.Selection import sel_e_greedy_fn
from trials.Selection import sel_middle_fn
from trials.Selection import sel_random_fn
from trials.Selection import sel_tournament_fn
from trials.Utilities import banner
from trials.Utilities import build_experiment_results
from trials.Utilities import get_spec
from trials.Utilities import random_spec
from trials.Utilities import write_table_html


def main():
    # 1 week of gpu time
    MAX_TIME = 60 * 60 * 24 * 7
    # number of best models to maintain
    N_BEST = 10
    # size of the population that should be maintained
    N_POP = 100
    # number of trials that should be performed with a single set of parameters
    # note trials are deterministic but may differ slightly due to multiple samples
    # present in the the nasbench dataset.
    N_EPOCH = 1
    # should we use halfway training values
    STOP_HALFWAY = True

    INITIAL_POPULATION = [
        random_spec(stop_halfway=STOP_HALFWAY) for idx in range(N_POP)
    ]

    BASE_TRIAL_ARGS = {
        "max_time": MAX_TIME,
        "num_best": N_BEST,
        "num_epochs": N_EPOCH,
    }

    def trial_name(text: str) -> str:
        if STOP_HALFWAY:
            return f"{text} Half"
        else:
            return f"{text} Full"

    def extend_trial(**kwargs) -> Dict:
        population = copy.deepcopy(INITIAL_POPULATION)
        return {**BASE_TRIAL_ARGS, "initial_population": population, **kwargs}

    trials: dict = dict()

    trials[trial_name("Baseline Random")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_random_fn(int(N_POP * 0.1)),
            "drp_fn": drop_random_fn(int(N_POP * 0.1)),
        }
    )
    trials[trial_name("Baseline Random Drop Worst")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_random_fn(int(N_POP * 0.1)),
            "drp_fn": drop_worst_fn(int(N_POP * 0.1)),
        }
    )
    trials[trial_name("Baseline Random Drop Oldest")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_random_fn(int(N_POP * 0.1)),
            "drp_fn": drop_oldest_fn(int(N_POP * 0.1)),
        }
    )

    for random_idx in range(1, 100):
        for select_idx in range(1, 100):

            if select_idx >= random_idx:
                continue

            n_random = int(random_idx)
            n_select = int(select_idx)
            n_remain = N_POP - n_select

            if n_remain < n_random:
                continue

            trials[
                trial_name(f"Tournament N-{int(random_idx)}% {int(select_idx)} Best")
            ] = extend_trial(
                **{
                    "mut_fn": mutate_fn(1.0),
                    "sel_fn": sel_tournament_fn(int(n_random), int(n_select)),
                    "drp_fn": drop_worst_fn(int(select_idx)),
                }
            )

    # for select_idx in range(1, 10):
    #     trials[trial_name(f"Greedy {int(select_idx*5)} Middle")] = extend_trial(
    #         **{
    #             "mut_fn": mutate_fn(1.0),
    #             "sel_fn": sel_middle_fn(int(select_idx * 5)),
    #             "drp_fn": drop_worst_fn(int(select_idx * 5)),
    #         }
    #     )

    # for select_idx in range(1, 100):
    #     for e_value in range(0, 100):
    #         e_value = e_value / 100
    #
    #         trials[
    #             trial_name(f"e-Greedy {int(e_value*100)}% e {int(select_idx)} Best")
    #         ] = extend_trial(
    #             **{
    #                 "mut_fn": mutate_fn(1.0),
    #                 "sel_fn": sel_e_greedy_fn(int(select_idx), float(e_value)),
    #                 "drp_fn": drop_worst_fn(int(select_idx)),
    #             }
    #         )

    # jury-rigged multi-threading, just re-run the program multiple times
    items = list(trials.items())
    random.shuffle(items)
    for name, params in items:
        path = os.path.join(os.getcwd(), "graphs", name)

        if os.path.exists(path):
            print(f"Skipping {name}, it already exists")
            continue

        banner(name)
        results = search(**params)

        for idx, [population, best, done, generations] in enumerate(results):
            epoch_path = os.path.join(path, f"Epoch {idx+1}")

            os.makedirs(epoch_path)

            pdf, bdf, adf, pdf_stats, bdf_stats, adf_stats = build_experiment_results(
                population, best, list(map(lambda x: get_spec(x), done))
            )

            write_table_html(pdf, epoch_path, "Population.html")
            write_table_html(bdf, epoch_path, "Best.html")
            write_table_html(adf, epoch_path, "All.html")
            write_table_html(pdf_stats, epoch_path, "PopulationStats.html")
            write_table_html(bdf_stats, epoch_path, "BestStats.html")
            write_table_html(adf_stats, epoch_path, "AllStats.html")

            pdf.to_csv(os.path.join(epoch_path, "Population.csv"))
            bdf.to_csv(os.path.join(epoch_path, "Best.csv"))
            adf.to_csv(os.path.join(epoch_path, "All.csv"))
            pdf_stats.to_csv(os.path.join(epoch_path, "PopulationStats.csv"))
            bdf_stats.to_csv(os.path.join(epoch_path, "BestStats.csv"))
            adf_stats.to_csv(os.path.join(epoch_path, "AllStats.csv"))

            generations_path = os.path.join(epoch_path, f"Heredity")
            os.makedirs(generations_path, exist_ok=True)
            for idx, generation in enumerate(generations):
                generation = [ind.get_hash() for ind in generation]
                json_path = os.path.join(generations_path, f"Generation{idx:06d}.json")

                with open(json_path, "w", encoding="utf8") as handle:
                    handle.write(json.dumps(generation))


if __name__ == "__main__":
    main()
