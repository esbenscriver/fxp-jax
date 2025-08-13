
import jax.numpy as jnp
from jax import random

from FixedPointJAX import FixedPointRoot

import pytest

@pytest.mark.parametrize("N, acceleration", [
    (1000, "None"),
    (1000, "SQUAREM"),
]) 

def test_FixedPointRoot(N: int, acceleration: str):

    a = random.uniform(random.PRNGKey(111), (N,1))
    b = random.uniform(random.PRNGKey(112), (1,1))

    def fxp(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        y = a + x @ b
        return y, y - x

    x, (step_norm, root_norm, iterations) = FixedPointRoot(
        fxp, 
        jnp.zeros_like(a).copy(),
        root_tol=1e-10, 
        acceleration=acceleration,
    )

    assert jnp.allclose(x, fxp(x)[0]), f"{jnp.linalg.norm(x-fxp(x)[0])}"