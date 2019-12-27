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
from cirq import ops, linalg

PAULI_BASIS = [op._unitary_() for op in [ops.I, ops.X, ops.Y, ops.Z]]


def schmidt_decomposition(U, atol=1e-9):
    """Calculates Schmidt decomposition of 4x4 unitary matrix.

    Represents unitary matrix U as linear combination of tensor products of 2x2 unitaries:
        U = Σ_i z_i * A_i ⊗ B_i,
    where A_i, B_i - 2x2 unitary matrices, z_i - positive and real, Σ_i |z_i|^2 = 1.
    Sum has 1, 2, or 4 terms.

    Args:
        U: Unitary matrix to decompose.
        atol: Ignore coefficients whose absolute value is smaller than this. Defaults to 1e-9.

    Returns:
        Dict with keys `first_qubit_ops`, `second_qubit_ops` and `koeffs`, containing values of
        A_i, B_i and z_i respectively.

    Reference: https://arxiv.org/pdf/1006.3412.pdf
    """
    assert U.shape == (4, 4)
    assert linalg.is_unitary(U)

    kak = linalg.kak_decomposition(U)
    c1, c2, c3 = [2 * c for c in kak.interaction_coefficients]
    B0, B1 = kak.single_qubit_operations_before
    A0, A1 = kak.single_qubit_operations_after
    g = kak.global_phase

    # Caculate coefficients.
    z = [
        0.5 * (np.exp(0.5j * c1) * np.cos(0.5 * (c3 - c2)) +
               np.exp(-0.5j * c1) * np.cos(0.5 * (c3 + c2))),
        0.5 * (np.exp(0.5j * c1) * np.cos(0.5 * (c3 - c2)) -
               np.exp(-0.5j * c1) * np.cos(0.5 * (c3 + c2))),
        -0.5j * (np.exp(0.5j * c1) * np.sin(0.5 * (c3 - c2)) -
                 np.exp(-0.5j * c1) * np.sin(0.5 * (c3 + c2))),
        0.5j * (np.exp(0.5j * c1) * np.sin(0.5 * (c3 - c2)) +
                np.exp(-0.5j * c1) * np.sin(0.5 * (c3 + c2))),
    ]

    # Throw away zero coefficients.
    take = [i for i in range(4) if abs(z[i]) > atol]
    z = [z[i] for i in take]
    a = [g * A0 @ PAULI_BASIS[i] @ B0 for i in take]
    b = [A1 @ PAULI_BASIS[i] @ B1 for i in take]

    # Make coefficients real.
    for i in range(len(z)):
        a[i] *= (z[i] / np.abs(z[i]))
        z[i] = np.abs(z[i])

    return {
        'first_qubit_ops': a,
        'second_qubit_ops': b,
        'koeffs': np.array(z),
    }
