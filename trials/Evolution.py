# Some of this code is modified from code from the NASBench package, the license of which is below.

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
from typing import Callable
from typing import List

from trials.Constants import OP_SPOTS
from trials.Constants import nasbench
from trials.ModelSpec import SpecWrapper


def crossover_spec(spec_a: SpecWrapper, spec_b: SpecWrapper, cx_prb: float = 0.5) -> SpecWrapper:
    assert spec_a.matrix.shape == spec_b.matrix.shape

    while True:
        new_matrix = copy.deepcopy(spec_a.matrix)
        new_ops = copy.deepcopy(spec_a.ops)

        for ridx, row in enumerate(spec_b.matrix):
            for cidx, itm in enumerate(row):
                if random.random() < cx_prb:
                    new_matrix[ridx][cidx] = itm

        for oidx, op in enumerate(spec_b.ops):
            if random.random() < cx_prb:
                new_ops[oidx] = op

        new_spec = SpecWrapper(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            return new_spec


def mutate_fn(mut_rate: float) -> Callable[[List[SpecWrapper]], List[SpecWrapper]]:
    assert mut_rate <= 1
    assert mut_rate >= 0

    def mutate_spec(old_spec: SpecWrapper):
        # failsafe, just in case, should never trigger in practice
        n_tries = 500
        old_size = old_spec.matrix.shape[0]
        while True:
            new_matrix = copy.deepcopy(old_spec.original_matrix)
            new_ops = copy.deepcopy(old_spec.original_ops)

            # In expectation, V edges flipped (note that most end up being pruned).
            edge_mutation_prob = mut_rate / old_size
            for src in range(0, old_size - 1):
                for dst in range(src + 1, old_size):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            # In expectation, one op is resampled.
            op_mutation_prob = mut_rate / OP_SPOTS
            for ind in range(1, old_size - 1):  # input/output is fixed
                if random.random() < op_mutation_prob:
                    available = [
                        o for o in nasbench.config["available_ops"] if o != new_ops[ind]
                    ]
                    new_ops[ind] = random.choice(available)

            new_spec = SpecWrapper(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return new_spec

            n_tries -= 1
            if n_tries < 0:
                raise Exception(f"Ran out of tries while mutating spec {old_spec}")

    def fn(population: List[SpecWrapper]) -> List[SpecWrapper]:
        for idx in range(len(population)):
            population[idx] = mutate_spec(population[idx])

        return population

    return fn



