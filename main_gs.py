import jax
import jax.numpy as jnp
import netket as nk
import time
from netket.experimental.driver.vmc_srt import VMC_SRt
from ansatz.transformer1d import Transformer_Enc

t0 = time.time()

seed = 0

# 1D Lattice

    
#### EDITS
if 1:
    L = 12
    J2 = 0.0

    diag_shift = 1e-3
    eta = 0.01
    N_opt = 300
    N_samples = 3000 # number monte carlo samples
    N_discard = 0

#####

# Settings wave function
f = 1
heads = 1
d_model = f * heads
b = 4

#! End input

lattice = nk.graph.Chain(length=L, pbc=True, max_neighbor_order=2)

# Hilbert space of spins on the graph
hilbert = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)

# Heisenberg spin hamiltonian
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J = [1.0, J2], sign_rule=[False, False])

# compute the ground-state energy
if (L <= 16):
    evals = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=False)
    print('The exact ground-state energy is E0 = ', evals[0])
    exact_gs_energy = evals[0]
else:
    exact_gs_energy = -0.4438 * L

# Variational wave function
wf = Transformer_Enc(d_model=d_model,
                    h=heads,
                    L=L//b,
                    b=b)

key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key, num=2)

print('\n key=\n',key)
print('\n subkey = \n', subkey)

params = wf.init(subkey, jnp.zeros((1,lattice.n_nodes)))

#print('\n params = \n', params)

#print('\n params = \n', params)
#print('\n hilbert = \n', hilbert)

init_samples = jnp.zeros((1,))

# Metropolis Local Sampling
sampler = nk.sampler.MetropolisExchange(hilbert=hilbert,
                                        graph=lattice,
                                        d_max=2,
                                        n_chains=N_samples,
                                        sweep_size=lattice.n_nodes)

key, subkey = jax.random.split(key, 2)
sampler_seed = subkey

vstate = nk.vqs.MCState(sampler=sampler, 
                        model=wf, 
                        sampler_seed=sampler_seed,
                        n_samples=N_samples, 
                        n_discard_per_chain=N_discard,
                        variables=params)

# vstate = nk.vqs.MCState(sampler=sampler, 
#                         model=wf, 
#                         n_samples=N_samples, 
#                         n_discard_per_chain=N_discard)

print('Number of parameters = ', nk.jax.tree_size(vstate.parameters))

# Variational monte carlo driver
optimizer = nk.optimizer.Sgd(learning_rate=eta)

if 0:
    vmc = VMC_SRt(hamiltonian=hamiltonian,
                optimizer=optimizer,
                diag_shift=diag_shift,
                variational_state=vstate)
    
sr = nk.optimizer.SR(diag_shift=0.1)

vmc = nk.VMC(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    preconditioner=sr,
    variational_state=vstate)




JI = vstate.parameters['encoder']['JI']
print('JI shape = ', JI.shape)

JR = vstate.parameters['encoder']['JR']
print('JR shape = ',JR.shape)

W0I = vstate.parameters['encoder']['W0I']['kernel']
print('W0I (kernel) shape = ',W0I.shape)

W0R = vstate.parameters['encoder']['W0R']['kernel']
print('W0R (kernel) shape = ',W0R.shape)

v_projR = vstate.parameters['encoder']['v_projR']['kernel']
print('v_projR (kernel) shape = ',v_projR.shape)

v_projI = vstate.parameters['encoder']['v_projI']['kernel']
print('v_projI (kernel) shape = ',v_projI.shape)

v_projR_bias = vstate.parameters['encoder']['v_projR']['bias']
print('v_projR (bias) shape = ',v_projR_bias.shape)
v_projI_bias = vstate.parameters['encoder']['v_projI']['bias']
print('v_projI (bias) shape = ',v_projI_bias.shape)
W0I_bias = vstate.parameters['encoder']['W0I']['bias']
print('W0I (bias) shape = ',W0I_bias.shape)
W0R_bias = vstate.parameters['encoder']['W0R']['bias']
print('W0R (bias) shape = ',W0R_bias.shape)


print('\n W0R = ', W0R)

# print(v_projR)


if 1:
    # Optimization
    start = time.time()
    vmc.run(out = 'ViT', n_iter = N_opt)
    end = time.time()
    # Import Json, this will be needed to load log files
    import json
    import matplotlib.pyplot as plt

    # import the data from log file
    data = json.load(open("ViT.log"))

    # Extract the relevant information
    iters = data["Energy"]["iters"]
    energy_Re = data["Energy"]["Mean"]["real"]


    t=time.time()
    print('\n Runtume: ', (t-t0))

    if 1:
        fig, ax1 = plt.subplots()
        ax1.plot(iters, energy_Re,color='C8', label='Energy (ViT)')

        ax1.set_ylabel('Energy')
        ax1.set_xlabel('Iteration')
        ax1.set_title('L=%i, J2=%.1f, b=%i, heads=%i'%(L,J2,b,heads))
        #plt.axis([0,iters[-1],exact_gs_energy-0.03,exact_gs_energy+0.2])
        plt.axhline(y=exact_gs_energy, xmin=0, xmax=iters[-1], linewidth=2, color='k', label='Exact')
        ax1.legend()
        plt.show()

    #print('\n iters \n', iters)
    #print('\n energy_Re \n', energy_Re)



    print('\n system size \n L=', L)
    print('\n num lattice nodes \n ', lattice.n_nodes)


#print('\n JI = ', JI)
#print(JI.shape)





