import torch as tn
import torchtt as tntt
import TTCME
import matplotlib.pyplot as plt 
import datetime
import numpy as np
import pickle 

tn.set_default_tensor_type(tn.DoubleTensor)

r1m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->', 'c1', params = ['c1'])
r2m = TTCME.ChemicalReaction(['mRNA','protein'],'mRNA->mRNA+protein', 'c2', params = ['c2'])
r3m = TTCME.ChemicalReaction(['mRNA','protein'],'->mRNA', 'c3', params = ['c3'])
r4m = TTCME.ChemicalReaction(['mRNA','protein'],'protein->', 'c4', params = ['c4'])

mdl = TTCME.ReactionSystem(['mRNA','protein'],[r1m, r2m, r3m, r4m], params= ['c1','c2','c3','c4'])

rates = np.array([0.002,0.015,0.1,0.01])
IC = [2,4]
N = [64,64]


#%% Get observation
np.random.seed(34548)


# reaction_time,reaction_jumps,reaction_indices = Gillespie(np.array(Initial),time_observation[-1],Pre,Post-Pre,rates)
# observations = Observations_grid(time_observation, reaction_time, reaction_jumps)
# observations_noise = observations+np.random.normal(0,sigma,observations.shape)

with open(r"simplegene_64_500k.pickle", "rb") as input_file:
    dct = pickle.load(input_file) 

No = dct['time_observation'].size
time_observation = dct['time_observation']
reaction_time = dct['reaction_time']
reaction_jumps = dct['reaction_jumps']
reaction_indices = dct['reaction_indices']
observations = dct['observations']
observations_noise = dct['observations_noise']
dT = time_observation[1]-time_observation[0]
sigma = dct['sigma']
sample_posterior_mcmc = dct['sample']



def identify_params(Nl,mult = 6,eps_solver=1e-5,ntmax=8,deg = 2,qtt = True):

    # basis parameters
    param_range = [[0,rc*mult] for rc in rates]
    basis_param = [TTCME.basis.BSplineBasis(Nl,[p[0],p[1]],deg = deg) for p in param_range]
    
    # prior
    mu = rates
    var = rates / np.array([1000,350,25,600])
    alpha_prior = mu**2/var
    beta_prior = mu/var
    prior = TTCME.GammaPDF(alpha_prior, beta_prior, basis_param, ['c1','c2','c3','c4'])

    # IC
    p_ic = TTCME.pdf.SingularPMF(N,IC,['mRNA','protein'])
    p0 = p_ic ** prior
    p0.normalize()

    p = p0.dofs.clone()
        
    # observation operator
    obs_operator = TTCME.pdf.GaussianObservation(N, [sigma]*2)
    
    # CME operator
    Stt,Mtt,Mtt_inv = mdl.generator_tt_galerkin(N, basis_param)
    Att = Mtt_inv @ Stt

    if qtt:
        A_qtt = Att.to_qtt()
        fwd_int = TTCME.TimeIntegrator.TTInt(A_qtt, epsilon = eps_solver, N_max = ntmax, dt_max = 1.0,method='cheby')
        Nbs = 4
        p = p.to_qtt()
    else:
        fwd_int = TTCME.TimeIntegrator.TTInt(Att, epsilon = eps_solver, N_max = ntmax, dt_max = 1.0,method='crank–nicolson')
        Nbs = 4



    # print('Starting...')
    storage = 0
    posterior_list  =[]
    joint_pdf = p0.copy()
    for i in range(1,No):
        
        y = observations_noise[i,:]

        
        po = obs_operator.likelihood(y) 
        #po = po / po.sum()
        
        po = po ** tntt.ones([Nl]*4)

        if qtt: po = po.to_qtt()
        
        #print('new observation ',i,'/',No,' at time ',time_observation[i],' ',y)
        
        tme = datetime.datetime.now()
        p = fwd_int.solve(p, dT, intervals = Nbs,qtt = qtt,verb = False, rounding=True, device = 'cpu')
        tme = datetime.datetime.now() - tme
        
        
        
        #print('\tmax rank ',max(p.R))
        p_pred = p
        p_post = po * p_pred
        p_post = p_post.round(1e-10)
        #print('\tmax rank (after observation) ',max(p_post.R))
        
        if storage<tntt.numel(p_post): storage = tntt.numel(p_post)*8/1000000

        if qtt:
            joint_pdf.dofs = tntt.reshape(p_post,joint_pdf.dofs.N)
            Z = joint_pdf.Z
            joint_pdf.normalize()
            p = p_post / Z
        else:
            joint_pdf.dofs = p_post.clone()
            joint_pdf.normalize()
            p = joint_pdf.dofs.clone()

        
        posterior_pdf = joint_pdf.marginal([0,1])
        posterior_pdf.round(1e-10)
        posterior_list.append(posterior_pdf.copy())



        E = posterior_pdf.expected_value()
        
        #print('\tExpected value computed posterior ' ,E)
        #print('\tposterior size ',tntt.numel(p)*8 / 1000000,' MB')
        #print('\telapsed ',tme)
        
        
    
    posterior_pdf.normalize()


    E = np.array(posterior_pdf.expected_value())
    C = np.array(posterior_pdf.covariance_matrix())
    V = np.diag(C)

    return posterior_pdf, E, V, C, storage

#%% Sweeping



epsilons = [1e-3,1e-4,1e-5,1e-6]
Nls = [16,32,64,128]
ntmax = [4,8,16,32]
degs = [1,2,3,4]

nburn = 250000
mean = np.mean(sample_posterior_mcmc[nburn:,:],0)
var = np.std(sample_posterior_mcmc[nburn:,:],0)**2
print('MCMC mean', mean)

tme = datetime.datetime.now()
Pt_ref, E_ref, V_ref, C_ref, _ = identify_params(64,6,1e-7,16)
tme = datetime.datetime.now() - tme
print('\ntime fine grid',tme)
print()

Es1 = []
Vs1 = []
time1 = []
memory1 = []
for eps in epsilons:
    print('epsilon ',eps)
    tme = datetime.datetime.now()
    Pt, E, V, C, mem = identify_params(64,6,eps,8)
    tme = datetime.datetime.now() - tme
    print('\ttime ',tme)
    print('\tE ',E)
    print('\tV ',V)
    Es1.append(E)
    Vs1.append(V)
    time1.append(tme.total_seconds())
    memory1.append(mem)

errE1 = np.mean(np.abs(Es1 - mean)/np.outer(np.ones(4),mean),1)
errV1 = np.mean(np.abs(Vs1 - var)/np.outer(np.ones(4),var),1)
print('error w.r.t. MCMC expectation \n',errE1)
print('error w.r.t. MCMC variance\n',errV1)
errE1 = np.mean(np.abs(Es1 - E_ref)/np.outer(np.ones(4),mean),1)
errV1 = np.mean(np.abs(Vs1 - V_ref)/np.outer(np.ones(4),var),1)
print('error w.r.t. fine posterior expectation\n ',errE1)
print('error w.r.t. fine posterior variance\n',errV1)
print('times ',time1)
print('memory [MB]',memory1)

Es2 = []
Vs2 = []
time2 = []
memory2 = []
for Nl in Nls:
    print('ell ',Nl)
    tme = datetime.datetime.now()
    Pt, E, V, C, mem = identify_params(Nl,6,1e-6,8)
    tme = datetime.datetime.now() - tme
    print('\ttime ',tme)
    print('\tE ',E)
    print('\tV ',V)
    Es2.append(E)
    Vs2.append(V)
    time2.append(tme.total_seconds())
    memory2.append(mem)
    
errE2 = np.abs(Es2 - mean)/mean * 100
errV2 = np.abs(Vs2 - var)/var * 100
print('error w.r.t. MCMC expectation ',errE2)
print('error w.r.t. MCMC variance',errV2)
errE2 = np.abs(Es2 - E_ref)/mean * 100
errV2 = np.abs(Vs2 - V_ref)/var * 100
print('error w.r.t. fine posterior expectation ',errE2)
print('error w.r.t. fine posterior variance',errV2)
print('times ',time2)
print('memory [MB]',memory2)

Es3 = []
Vs3 = []
time3 = []
memory3 = []
for nt in ntmax:
    print('T ',nt)
    tme = datetime.datetime.now()
    Pt, E, V, C, mem = identify_params(64,6,1e-6,nt)
    tme = datetime.datetime.now() - tme
    print('\ttime ',tme)
    print('\tE ',E)
    print('\tV ',V)
    Es3.append(E)
    Vs3.append(V)
    time3.append(tme.total_seconds())
    memory3.append(mem)
   
errE3 = np.abs(Es3 - mean)/mean * 100
errV3 = np.abs(Vs3 - var)/var * 100
print('error w.r.t. MCMC expectation ',errE3)
print('error w.r.t. MCMC variance',errV3)
errE3 = np.abs(Es3 - E_ref)/mean * 100
errV3 = np.abs(Vs3 - V_ref)/var * 100
print('error w.r.t. fine posterior expectation ',errE3)
print('error w.r.t. fine posterior variance',errV3)
print('times ',time3)
print('memory [MB]',memory3)

Es4 = []
Vs4 = []
time4 = []
memory4 = []
for deg in degs:
    print('deg ',deg)
    tme = datetime.datetime.now()
    Pt, E, V, C, mem = identify_params(64,6,1e-5,8,deg=deg)
    tme = datetime.datetime.now() - tme
    print('\ttime ',tme)
    print('\tE ',E)
    print('\tV ',V)
    Es4.append(E)
    Vs4.append(V)
    time4.append(tme.total_seconds())
    memory4.append(mem)
   
errE4 = np.abs(Es4 - mean)/mean * 100
errV4 = np.abs(Vs4 - var)/var * 100
print('error w.r.t. MCMC expectation ',errE4)
print('error w.r.t. MCMC variance',errV4)
errE4 = np.abs(Es4 - E_ref)/mean * 100
errV4 = np.abs(Vs4 - V_ref)/var * 100
print('error w.r.t. fine posterior expectation ',errE4)
print('error w.r.t. fine posterior variance',errV4)
print('times ',time4)
print('memory [MB]',memory4)


#%%

Es1 = np.array(Es1)
Es2 = np.array(Es2)
Es3 = np.array(Es3)

Vs1 = np.array(Vs1)
Vs2 = np.array(Vs2) 
Vs3 = np.array(Vs3)
    
errE1 = np.abs(Es1 - mean)/mean * 100
errE2 = np.abs(Es2 - mean)/mean * 100
errE3 = np.abs(Es3 - mean)/mean * 100

errV1 = np.abs(Vs1 - var)/var * 100
errV2 = np.abs(Vs2 - var)/var * 100
errV3 = np.abs(Vs3 - var)/var * 100