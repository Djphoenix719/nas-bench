import copy
import os.path
from typing import Dict
import json

from trials.Evolution import mutate_fn
from trials.Search import search
from trials.Selection import drop_worst_fn
from trials.Selection import sel_random_fn
from trials.Selection import sel_tournament_fn
from trials.Utilities import banner
from trials.Utilities import build_experiment_results
from trials.Utilities import get_spec
from trials.Utilities import random_spec
from trials.Utilities import write_table_html


def main():
    # 1 month of GPU time
    MAX_TIME = 60 * 60 * 24 * 31 * 1
    # number of best models to maintain
    N_BEST = 10
    # size of the population that should be maintained
    N_POP = 250
    # number of trials that should be performed with a single set of parameters
    # note trials are deterministic but may differ slightly due to multiple samples
    # present in the the nasbench dataset.
    N_EPOCH = 10
    # should we use halfway training values
    STOP_HALFWAY = True

    INITIAL_POPULATION = [random_spec() for idx in range(N_POP)]

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

    trials[trial_name("Baseline Random Full")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_random_fn(int(N_POP * 0.1)),
            "drp_fn": drop_worst_fn(int(N_POP * 0.1)),
        }
    )
    trials[trial_name("Tournament N-10% 5 Best")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_tournament_fn(int(N_POP * 0.1), 5),
            "drp_fn": drop_worst_fn(5),
        }
    )
    trials[trial_name("Tournament N-20% 10 Best")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_tournament_fn(int(N_POP * 0.2), 10),
            "drp_fn": drop_worst_fn(10),
        }
    )
    trials[trial_name("Tournament N-40% 20 Best")] = extend_trial(
        **{
            "mut_fn": mutate_fn(1.0),
            "sel_fn": sel_tournament_fn(int(N_POP * 0.4), 20),
            "drp_fn": drop_worst_fn(20),
        }
    )

    for name, params in trials.items():
        banner(name)
        params["initial_population"] = [random_spec() for idx in range(N_POP)]
        results = search(**params)
        path = os.path.join(os.getcwd(), "graphs", name)

        for idx, [population, best, done, generations] in enumerate(results):
            epoch_path = os.path.join(path, f"Epoch {idx+1}")

            if os.path.exists(epoch_path):
                continue

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
