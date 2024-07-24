from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence
from jax._src import dtypes
import netket as nk
import time
import json
import matplotlib.pyplot as plt


class Transformer(nn.Module):
    L: int # chain length
    b: int # cluster length

    
    @nn.compact
    def __call__(self, x):
        Nc = self.L // self.b

        Q = self.param('Q', nn.initializers.normal(), (self.b, self.b), jnp.complex128)
        K = self.param('K', nn.initializers.normal(), (self.b, self.b), jnp.complex128)
        V = self.param('V', nn.initializers.normal(), (self.b, self.b), jnp.complex128)
        W = self.param('W', nn.initializers.normal(), (self.L, self.L), jnp.complex128)

        x = x.reshape(Nc, self.b)

        Qx = jnp.matmul(x, Q.T)
        Kx = jnp.matmul(x, K.T)
        Vx = jnp.matmul(x, V.T)

        z = jnp.matmul(Qx, Kx.T) / jnp.sqrt(self.L)

        alist = jax.nn.softmax(-jnp.diag(z))

        vtilde = (alist[:, jnp.newaxis] * Vx).reshape(self.L)
        return vtilde.T @ W @ vtilde
    

L = 12  # Linear size of the lattice
b = 2
hilbert = nk.hilbert.Spin(s=0.5, N=L)
# graph = Grid(length=L, pbc=True)/
lattice = nk.graph.Chain(length=L, pbc=True, max_neighbor_order=1)
# hamiltonian = Heisenberg(hilbert=hilbert, graph=graph)
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J = 1.0, sign_rule=[False]) # 

model = Transformer(L,b)

seed = 0
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key, num=2)
params = model.init(subkey, jnp.zeros((1,lattice.n_nodes)))
init_samples = jnp.zeros((1,))
key, subkey = jax.random.split(key, 2)
sampler_seed = subkey

sampler = nk.sampler.MetropolisLocal(hilbert)
vstate = nk.vqs.MCState(sampler=sampler, model=model, variables=params, n_samples=1008)

x = jnp.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
# Evaluate the logarithm of the wavefunction amplitude for the specific configuration x
log_val = vstate.log_value(x)

# Convert to the wavefunction coefficient by exponentiating the log value
psi_x = jnp.exp(log_val)

print(f"Wavefunction coefficient for x={x}: {psi_x}")