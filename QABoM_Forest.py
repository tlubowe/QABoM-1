import pyquil.quil as pq
import pyquil.api as api
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
import json

import copy
from scipy.optimize import minimize



import numpy as np
import math
import random

gate_noise_probs = [0.001,0.001,0.001]
meas_noise_probs = [0.001,0.001,0.001]

#qvm = api.SyncConnection('http://127.0.0.1:5000', gate_noise=gate_noise_probs, measurement_noise=meas_noise_probs)
qvm = api.QVMConnection()

from grove.pyvqe.vqe import VQE
vqe_inst = VQE(minimizer=minimize,
               minimizer_kwargs={'method': 'nelder-mead'})

beta = 1.0
state_prep_beta = np.arctan(math.e**(-beta/2.0)) * 2.0

n_qubits = 8
num_good_qubits = int(n_qubits / 2)

n_qaoa = 2
#use None value for "god mode" expectation.
# must be greater than number of data samples.
n_measurement_samples = None

def classical_sample(beta):
    """
    Classical weighted coin based on the thermal distribution of the initial
    state.   
    
    Input:
        beta: Inverse temperature to the initial qubits
    Output:
        Weighted coin on whether to perform a qubit flip
    """
    p = 0.5*(np.tanh(beta)+1)
    if random.random() < p:
        return 0
    else:
        return 1
    
def state_prep_classical(beta_value, start_qb):
    """
    Alternative state prep. Effectively creates an initial thermal state by clasically sampling a distribution
    and feeding that into the circuit.  rho_m = exp(-beta*H_M)/Z_m
    
    Input:
        beta_value: Inverse temperature to the initial qubits
        start_qb: Initialize on a specific qubit number to avoid dead qubits on
                  Rigetti
    Output:
        state_prep: A Rigetti Quil program for the initial state
        
    -------X^(classical outcome on qubit j)----- H -------
    """
    
    state_prep = pq.Program()
    for i in range(start_qb, num_good_qubits + start_qb,1):
        tmp = pq.Program()
        samp = classical_sample(beta_value)
        if samp == 0:
            tmp.inst(H(i))
        else:
            tmp.inst(X(i),H(i))
        state_prep = state_prep + tmp
    return state_prep
    

def state_prep(beta_value, start_qb=0):
    """
    Prepares initial thermal state by using extra qubits and entanglement. 
    Reduced state on "good qubits" is rho_m = exp(-beta*H_M)/Z_m.
    
    Input:
        beta_value: Inverse temperature to initial qubits to
        start_qb: Initialize on specific qubit number to avoid dead qubits on
                  Rigetti
    Output:
        state_prep: A Rigetti Quil program for the initial state
    """

    state_prep = pq.Program()
    for i in range(start_qb, num_good_qubits + start_qb, 1):
            tmp = pq.Program()
            tmp.inst(RX(beta_value, i), CNOT(i,i + num_good_qubits), S(num_good_qubits + i), H(num_good_qubits + i))
            state_prep = state_prep + tmp

    return state_prep

def full_mixer_hamiltonian(nu_value,start_qb=0):
    """
    Implement exponential of full mixer Hamiltonian across all qubits.
    
    Input:
        nu_value:  pulse parameter
        start_qb: Initialized on specific qubit number to avoid dead qubits on
                  Rigetti
    Output:
        Hi: A Rigetti Quil program for the mixer Hamiltonian    
    """
    
    Hi = pq.Program()
    for i in range(start_qb, num_good_qubits + start_qb, 1):
        tmp = pq.Program(RX(2.0 * nu_value, i))
        Hi = Hi + tmp

    return Hi

def biases_cost(nu_value, bias, start_qb=0):
    """
    Implements exponential of 1-local terms (Z) of full cost Hamiltonian.
    
    Input:
        nu_value: pulse parameter
        bias: bias value
        start_qb: which qubit to start on (either the first qubit or first
                  hidden qubit)
    Output:
        BiasC: Quil Program associated with local bias energy cost    
    """

    assert(len(bias) <= num_good_qubits)

    BiasC = pq.Program()
    for i in range(start_qb, num_good_qubits + start_qb, 1):
        tmp = pq.Program(RZ(2.0 * nu_value * bias[i-start_qb], i))
        BiasC = BiasC + tmp

    return BiasC

def weights_cost(nu_value, weights):
    """
    Implements exponential of 2-local terms (ZZ) of full cost Hamiltonian.
    
    Input:
        nu_value: pulse parameter
        weights: J_jk weights across all qubit pairs
    Output:
        weights_cost: Quil program creating weight cost Hamiltonian
    """

    assert(max(len(weights), len(weights[0])) <= num_good_qubits)

    weights_cost = pq.Program()
    for j in range(0, len(weights), 1):
        for k in range(0, len(weights[0]), 1):
            tmp = pq.Program()
            if j != k:
                tmp.inst(CNOT(j, k), RZ(2.0 * nu_value * weights[j][k], k), CNOT(j, k))
            else:
                tmp.inst(RZ(2.0 * nu_value * weights[j][k], k))
            weights_cost = weights_cost + tmp

    return weights_cost

def total_cost(nu_value, weights, bias, start_qb=0):
    """
     Implements exponential of full Hamiltonian for both clamped and unclamped programs
     
     Input:
         nu_value: pulse parameter
         weights: J_jk weights for the program
         bias: B_j biases for the program
         start_qb: Starting position for the program (either first qb or first
                   hidden qubit)
     Output:
         Hf: Quil program for the total cost Hamiltonian
    """

    part1 = biases_cost(nu_value, bias, start_qb=start_qb)
    part2 = weights_cost(nu_value, weights)

    Hf = part1 + part2
    return Hf

####CLAMPED SAMPLING PART!!!!
def clamped_state_prep(beta_value, data_vector):
    """
    Make the clamped initial state. 
    
    Input:
        beta_value: Inverse temperature
        data_vector: bit string for data point to be clamped 
    Output:
        Clamped state prep program
    """
    
    v_len = len(data_vector)

    clamped_start_state = state_prep(beta_value, start_qb=v_len)

    data_prog = pq.Program()
    for i in range(v_len):
        if int(data_vector[i]) == 1:
            data_prog.inst(X(i))

    return clamped_start_state + data_prog

def partial_mixer(nu_value, data_vector):
    """
    Implements exponential of partial mixer Hamiltonian across the hidden layer
    
    Input:
        nu_value: pulse parameter
        data_vector: data point bit string
    Output:
        Quil Program of the partial mixer Hamiltonian
    """
    
    v_len = len(data_vector)
    return full_mixer_hamiltonian(nu_value, start_qb=v_len)

def clamped_total_cost(nu_value, data_vector, weights, bias):
    """
    Make clamped Total Cost function starting on first hidden qubit
    
    Input:
        nu_value: pulse parameter
        data_vector: data point bit string
        weights: J_jk values for the 2-local weight costs
        bias: B_j values for the local bias costs
    Output:
        Clamped total cost
    """
    v_len = len(data_vector)
    return total_cost(nu_value,weights, bias, start_qb=v_len)

def make_clamped_prog(var_vector, data_vector, weights, bias):
    """
    Make total clamped program
    
    Input:
        var_vector: QAOA pulse parameters vector
        data_vector: datapoint bitstring
        weights: J_jk for 2-local weight costs
        bias: B_j for local bias costs
    Output:
        big_program: Quil program from clamped state prep to cost
    """
    nu_vector = var_vector[:len(var_vector)/2]
    gamma_vector = var_vector[len(var_vector)/2:]

    big_program = clamped_state_prep(state_prep_beta, data_vector)
    for i in range(0, 2 * n_qaoa, 1):
        if i%2==0:
            big_program = big_program + clamped_total_cost(nu_vector[i/2],
                                                          data_vector,
                                                          weights,
                                                          bias)
        else:
            big_program = big_program + partial_mixer(gamma_vector[i/2],
                                                     data_vector)

    return big_program

#make full program
def make_regular_prog(var_vector, weights, bias):
    """
    Make total unclamped program
    
    Input:
        var_vector: QAOA pulse parameters vector
        weights: J_jk weights for 2-local weight costs
        bias: B_j biases for local bias costs        
    Output:
        big_program: Total unclamped Quil program
    """
    nu_vector = var_vector[:len(var_vector)/2]
    gamma_vector = var_vector[len(var_vector)/2:]

    big_program = state_prep(state_prep_beta)
    for i in range(0, 2 * n_qaoa, 1):
        if i%2==0:
            big_program = big_program + total_cost(nu_vector[i/2],
                                                  weights,
                                                  bias)
        else:
            big_program = big_program + full_mixer_hamiltonian(gamma_vector[i/2])

    return big_program


def make_measurement_hamiltonian(weights, bias):
    """
    Makes the cost Hamiltonian operator to be measured.
    
    Inputs:
        weights: J_jk weights for the 2-local weight terms
        bias: B_j biases for the local bias terms
    Output:
        Meas_Ham: Quil program to create a measurement Hamiltonian to compute
                  expectations of
        
    """
    Meas_Ham = ID() - ID()
    for j in range(len(bias)):
        Meas_Ham = Meas_Ham + float(bias[j]) * sZ(j)


    for j in range(len(weights)):
        for k in range(len(weights[0])):
            Meas_Ham = Meas_Ham + float(weights[j][k]) * (sZ(j) * sZ(k))

    return Meas_Ham



def make_fxn_to_optimize(weights, bias):
    """
    Make unclamped optimization program and compute the expectation value
    
    Input:
        weights: J_jk weights for 2-local cost
        bias: B_j biases for local cost
        
    Output: 
        function object
    """

    def F_TO_OPT(var_vector):
        """
        Closed function to compute the expectation value of the program
        to find optimal energy parameters
        
        Input:
            var_vector: QAOA pulse parameters
        Output:
            Expectation value of the measurement hamiltonian for the given
            initial state
        """
        initial_state = make_regular_prog(var_vector, weights, bias)
        measure_ham = make_measurement_hamiltonian(weights, bias)

        return vqe_inst.expectation(initial_state,
                                    measure_ham,
                                    n_measurement_samples,
                                    qvm)
    return F_TO_OPT

def make_clamped_fxn_to_optimize(data_matrix, weights, bias):

    """
    Make clamped optimization program and compute the expectation value
    
    Input:
        weights: J_jk weights for 2-local cost
        bias: B_j biases for local cost
        
    Output: 
        function object
    """
    def F_TO_OPT(var_vector):
        """
        Closed function to compute the expectation value of the program
        to find optimal energy parameters
        
        Input:
            var_vector: QAOA pulse parameters
        Output:
            Expectation value of the measurement Hamiltonian for the given
            clamped initial state
        """

        exp_values = []

        if n_measurement_samples == None:
            for data_vector in data_matrix:
                initial_state = make_clamped_prog(var_vector,
                                                  data_vector,
                                                  weights,
                                                  bias)
                measure_ham = make_measurement_hamiltonian(weights, bias)
                
                exp_values.append(vqe_inst.expectation(initial_state,
                                                       measure_ham,
                                                       n_measurement_samples,
                                                       qvm))
            return float(sum(exp_values)) / float(len(exp_values))
        else:
            for data_vector in data_matrix:
                initial_state = make_clamped_prog(var_vector,
                                                  data_vector,
                                                  weights,
                                                  bias)
                measure_ham = make_measurement_hamiltonian(weights, bias)
                
                exp_values.append(vqe_inst.expectation(initial_state,
                                                       measure_ham,
                                                       n_measurement_samples / len(data_matrix),
                                                       qvm))
            return float(sum(exp_values)) / float(len(exp_values))
    return F_TO_OPT

import math
def sigmoid(x):
    return map(lambda a: 1./(1. + math.e**(-a)), x)


import time

#initial weights
RBM_WEIGHTS = np.asarray([[0.01, 0.01], [0.02, 0.4], [0.03, -0.01], [0.07, 0.03]])

#initial biases
RBM_BIAS = np.asarray([0.02,-0.02, 0.02, 0.02])

#data (see paper for data prep method)
DATA = np.asarray([[-1, -1,-1,-1],
                   [-1,-1, -1, -1],
                   [1, 1, -1, -1],
                   [1, 1, -1, -1],
                   [-1, -1, 1, 1],
                   [-1, -1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]])

orig_weights = copy.deepcopy(RBM_WEIGHTS)
orig_bias = copy.deepcopy(RBM_BIAS)

with open("RBM_info.txt", "r") as myfile:
    RBM_WEIGHTS = np.asarray(json.loads(myfile.readline()))
    RBM_BIAS = np.asarray(json.loads(myfile.readline()))


n_classical_opt_iter = 100 
n_epochs = 30 
learning_rate = 0.06 


for epoch in range(n_epochs):

    new_weights = copy.deepcopy(RBM_WEIGHTS)
    new_bias = copy.deepcopy(RBM_BIAS)

    #update weights
    minimal_energy_config_for_cur_weights = make_fxn_to_optimize(RBM_WEIGHTS,
                                                                 RBM_BIAS)
    optimal_energy_params = minimize(minimal_energy_config_for_cur_weights,
                                     np.random.rand(2 * n_qaoa),
                                     method='Nelder-Mead',
                                     tol=1e-3,
                                     options={'maxiter': n_classical_opt_iter})
    
    print( 'Found minimal energy configuration for given model weights....')

    initial_state = make_regular_prog(optimal_energy_params.x,
                                      RBM_WEIGHTS,
                                      RBM_BIAS)

    for a in range(len(RBM_WEIGHTS)):
        for b in range(len(RBM_WEIGHTS[0])):


            print( 'Processing weights[',a,'][',b,']...')


            model_expect_weight = vqe_inst.expectation(initial_state,
                                                sZ(a) * sZ(b),
                                                n_measurement_samples,
                                                qvm)
            model_expect_bias = vqe_inst.expectation(initial_state,
                                                     sZ(a + len(DATA[0])),
                                                     n_measurement_samples,
                                                     qvm)

            clamped_sum_weights = 0.0
            clamped_sum_bias = 0.0
            #Can avoid going through every point in data by clamping random data points
            print( 'Attempting qram-like measuremant of all data....')

            minimal_energy_config_for_cur_weights_clamped = make_clamped_fxn_to_optimize(DATA,
                                                                                         RBM_WEIGHTS,
                                                                                         RBM_BIAS)
            optimal_clamped_energy_params = minimize(minimal_energy_config_for_cur_weights_clamped,
                                                     np.random.rand(2 * n_qaoa),
                                                     method='Nelder-Mead',
                                                     tol=1e-3,
                                                     options={'maxiter': n_classical_opt_iter})
            optimal_clamped_energy_vector = optimal_clamped_energy_params.x

            #extracting expectation and update biases/weights
            data_state = make_clamped_prog(optimal_clamped_energy_vector,
                                           DATA[0],
                                           RBM_WEIGHTS,
                                           RBM_BIAS)

            data_expect_weight = vqe_inst.expectation(data_state,
                                               sZ(a) * sZ(b),
                                               n_measurement_samples,
                                               qvm) 

            data_expect_bias = vqe_inst.expectation(data_state,
                                                    sZ(a + len(DATA[0])),
                                                    n_measurement_samples,
                                                    qvm)

            clamped_sum_bias += data_expect_bias
            clamped_sum_weights += data_expect_weight

            # clamped_sum_bias /= float(len(DATA))
            # clamped_sum_weights /= float(len(DATA))
#         weight bias update
            new_weights[a][b] += learning_rate * (clamped_sum_weights - model_expect_weight)

            if b == 0:
                new_bias[a] += learning_rate * (clamped_sum_bias - model_expect_bias)


    RBM_WEIGHTS = copy.deepcopy(new_weights)
    RBM_BIAS = copy.deepcopy(new_bias)


    with open("RBM_info.txt", "w") as myfile:
        myfile.write(json.dumps(list(RBM_WEIGHTS.tolist()))+'\n')
        myfile.write(json.dumps(list(RBM_BIAS.tolist()))+'\n')

    with open("RBM_history.txt", "a") as myfile:
        myfile.write(json.dumps(list(RBM_WEIGHTS.tolist()))+'\n')
        myfile.write(json.dumps(list(RBM_BIAS.tolist()))+'\n')
        myfile.write(str('-'*80) + '\n')


    print ('SAVED!')
    print ('-' * 80)

print( 'Done!')

