# Copyright 2019 The Cirq Developers
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

import numpy as np
from scipy.stats import unitary_group
import cirq
from cirq.linalg import PAULI_BASIS
from cirq.linalg.predicates import is_unitary


def _check_schmidt_decomposition(U, expected_schmidt_number=None):
    sd = cirq.schmidt_decomposition(U)
    a = sd['first_qubit_ops']
    b = sd['second_qubit_ops']
    k = sd['koeffs']
    n = len(k)

    assert (n == 1 or n == 2 or n == 4)
    assert len(a) == n
    assert len(b) == n
    for i in range(n):
        assert is_unitary(a[i])
        assert is_unitary(b[i])
        assert np.allclose(k[i], np.abs(k[i]))
    assert np.allclose(np.linalg.norm(k), 1)

    if expected_schmidt_number is not None:
        assert n == expected_schmidt_number

    U_restored = sum([k[i] * np.kron(a[i], b[i]) for i in range(len(k))])
    assert np.allclose(U, U_restored)


def test_cnot():
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    _check_schmidt_decomposition(CNOT, expected_schmidt_number=2)


def test_pauli_matrices():
    for mx1 in PAULI_BASIS.values():
        for mx2 in PAULI_BASIS.values():
            _check_schmidt_decomposition(np.kron(mx1, mx2),
                                         expected_schmidt_number=1)


def test_random_unitaries():
    for _ in range(50):
        U = unitary_group.rvs(4)
        _check_schmidt_decomposition(U)


def test_random_unitaries_tesnor_products():
    for _ in range(50):
        U = np.kron(unitary_group.rvs(2), unitary_group.rvs(2))
        _check_schmidt_decomposition(U, expected_schmidt_number=1)
