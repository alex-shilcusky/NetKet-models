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
        # print('\n x = ',x )
        x = jnp.atleast_2d(x)
        return jax.vmap(self.evaluate_single, in_axes=(0))(x)
        
    def evaluate_single(self, x):
        Nc = self.L // self.b

        Q = self.param('Q', nn.initializers.normal(stddev=1), (self.b, self.b), jnp.complex128)
        K = self.param('K', nn.initializers.normal(stddev=1), (self.b, self.b), jnp.complex128)
        V = self.param('V', nn.initializers.normal(stddev=1), (self.b, self.b), jnp.complex128)
        W = self.param('W', nn.initializers.normal(stddev=1), (self.L, self.L), jnp.complex128)

        x = x.reshape(Nc, self.b)

        Qx = jnp.matmul(x, Q.T)
        Kx = jnp.matmul(x, K.T)
        Vx = jnp.matmul(x, V.T)

        z = jnp.matmul(Qx, Kx.T) / jnp.sqrt(self.L)

        alist = jax.nn.softmax(-jnp.diag(z))

        vtilde = (alist[:, jnp.newaxis] * Vx).reshape(self.L)
        return vtilde.T @ W @ vtilde
    

L = 16
b = 2

diag_shift = 0.001
eta = 0.01
N_opt = 300
N_samples = 3000 # number monte carlo samples
N_discard = 0

lattice = nk.graph.Chain(length=L, pbc=True, max_neighbor_order=2)
hilbert = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)

# hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J = 1.0, sign_rule=False) # 
# hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J = [1.0, 0.4], sign_rule=[False, False]) 
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J = [1.0, 0], sign_rule=[False, False]) 

print('\n Lattice is bipartite? ')
print(lattice.is_bipartite())

if (L <= 16):
    evals = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=False)
    exact_gs_energy = evals[0]
else:
    exact_gs_energy = -0.4438 * L * 4
print('The exact ground-state energy is E0 = ', exact_gs_energy)

wf = Transformer(L, b)

seed = 0
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key, num=2)
params = wf.init(subkey, jnp.zeros((1,lattice.n_nodes)))
init_samples = jnp.zeros((1,))
key, subkey = jax.random.split(key, 2)
sampler_seed = subkey

sampler = nk.sampler.MetropolisExchange(hilbert=hilbert,
                                        graph=lattice,
                                        d_max=L,
                                        n_chains=N_samples,
                                        sweep_size=lattice.n_nodes)
vstate = nk.vqs.MCState(sampler=sampler, 
                        model=wf, 
                        sampler_seed=sampler_seed,
                        n_samples=N_samples, 
                        n_discard_per_chain=N_discard,
                        variables=params)

print('Number of parameters = ', nk.jax.tree_size(vstate.parameters))
optimizer = nk.optimizer.Sgd(learning_rate=eta)
sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=True) # need small diag_shift for Transformer 

vmc = nk.VMC(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    preconditioner=sr,
    variational_state=vstate)


##################



if 1:
    # Optimization
    start = time.time()
    vmc.run(out = 'Transformer', n_iter = N_opt)
    end = time.time()

    # import the data from log file
    data = json.load(open("Transformer.log"))

    # Extract the relevant information
    iters = data["Energy"]["iters"]
    energy_Re = data["Energy"]["Mean"]["real"]

    fig, ax1 = plt.subplots()
    ax1.plot(iters, energy_Re,color='C8', label='Energy (ViT)')

    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    ax1.set_title('L=%i, b=%i '%(L,b))
    #plt.axis([0,iters[-1],exact_gs_energy-0.03,exact_gs_energy+0.2])
    plt.axhline(y=exact_gs_energy, xmin=0, xmax=iters[-1], linewidth=2, color='k', label='Exact')
    ax1.legend()
    plt.show()


if 0: 
    local_energies = vstate.local_estimators(hamiltonian)
    print(local_energies.shape)

    E = vstate.expect(hamiltonian)

    print('\n E = ', E)

    print('\n mean of local energies = ', jnp.mean(local_energies))
    E = vstate.expect(hamiltonian)

    # vstate.sample()
    # print(sampler.acceptance)
    # print(sampler)

    samples = vstate.samples
    print(samples[0])
    print(local_energies[0])

    plt.plot(range(N_samples), local_energies)
    plt.xlabel('Monte Carlo Iteration')
    plt.ylabel('Energy')
    plt.title('Energy vs. Monte Carlo Iteration')
    plt.show()



#print('\n iters \n', iters)
#print('\n energy_Re \n', energy_Re)
# print('\n system size \n L=', L)
# print('\n num lattice nodes \n ', lattice.n_nodes)


# params = vstate.parameters
# print('\n params = \n', params)
import numpy as np
Q = np.asarray(vstate.parameters['Q'])
K = vstate.parameters['K']
V = vstate.parameters['V']
W = vstate.parameters['W']

if 0:
    print('\n Q = \n', Q)
    print('\n K = \n', K)
    print('\n V = \n', V)
    print('\n W = \n', W)



