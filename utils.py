import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import cantera as ct


def split_exp_group(exp_XFO, exp_TP, exp_idt):
    group_indices = [0]
    P0 = exp_TP[0, 1]
    XFO0 = exp_XFO[0]
    for i in range(len(exp_idt)):
        P = exp_TP[i, 1]
        XFO = exp_XFO[i]
        if P < P0 - 5 or P > P0 + 5 or i == len(exp_idt) - 1 or XFO != XFO0:
            if i == len(exp_idt) - 1:
                group_indices.append(i + 1)
            else:
                group_indices.append(i)
            P0 = P
            XFO0 = XFO
    return group_indices


def load_st_idt(data_paths, mech_path):
    exp_data = []
    for data_file in data_paths:
        f = pd.read_csv(data_file)
        exp_data += f.to_dict('records')

    exp_X = []
    exp_TP = []
    exp_idt = []

    for i in range(len(exp_data)):
        m = {}
        for key in exp_data[i].keys():
            if key in ['NH3', 'H2', 'O2', 'N2', 'AR'] and exp_data[i][key] > 0:
                m[key] = exp_data[i][key]
        exp_X.append(m)
        exp_TP.append([exp_data[i]['T'], exp_data[i]['P']])
        exp_idt.append(exp_data[i]['t'])

    exp_phi = []
    for i in range(len(exp_data)):
        gas = ct.Solution(mech_path)
        gas.TP = exp_TP[i][0], exp_TP[i][1] * ct.one_atm
        gas.X = exp_X[i]
        exp_phi.append(gas.equivalence_ratio())

    exp_XFO = []
    for X in exp_X:
        m = {'fuel': {}, 'oxidizer': {}}
        for key in X.keys():
            if key in ['NH3', 'H2']:
                m['fuel'][key] = X[key]
            else:
                m['oxidizer'][key] = X[key]
        exp_XFO.append(m)

        exp_TP = np.array(exp_TP)
        exp_phi = np.array(exp_phi)
        exp_idt = np.array(exp_idt)

    return exp_XFO, exp_TP, exp_phi, exp_idt


def load_rcm_idt(data_paths, mech_path):
    exp_data = []
    for data_file in data_paths:
        f = pd.read_csv(data_file)
        exp_data += f.to_dict('records')

    exp_X = []
    exp_TP0 = []
    exp_TPeoc = []
    exp_TPeff = []
    exp_idt = []

    for i in range(len(exp_data)):
        m = {}
        for key in exp_data[i].keys():
            if key in ['NH3', 'H2', 'O2', 'N2', 'AR'] and exp_data[i][key] > 0:
                m[key] = exp_data[i][key]
        exp_X.append(m)
        if exp_data[i]['T0']:
            exp_TP0.append([exp_data[i]['T0'], exp_data[i]['P0']])
        else:
            exp_TPeoc.append([exp_data[i]['Teoc'], exp_data[i]['Peoc']])

        exp_TPeff.append([exp_data[i]['Teff'], exp_data[i]['Peff']])
        exp_idt.append(exp_data[i]['t'])

    exp_phi = []
    for i in range(len(exp_data)):
        gas = ct.Solution(mech_path)
        gas.X = exp_X[i]
        exp_phi.append(gas.equivalence_ratio())

    exp_XFO = []
    for X in exp_X:
        m = {'fuel': {}, 'oxidizer': {}}
        for key in X.keys():
            if key in ['NH3', 'H2']:
                m['fuel'][key] = X[key]
            else:
                m['oxidizer'][key] = X[key]
        exp_XFO.append(m)

        exp_TP0 = np.array(exp_TP0)
        exp_TPeoc = np.array(exp_TPeoc)
        exp_TPeff = np.array(exp_TPeff)
        exp_phi = np.array(exp_phi)
        exp_idt = np.array(exp_idt)

    return exp_XFO, exp_TP0, exp_TPeoc, exp_TPeff, exp_phi, exp_idt


def modify_A(gas, normA, UF):
    i = 0
    for idx in range(len(gas.reactions())):
        rxn = gas.reactions()[idx]
        if type(rxn.rate) == ct.ArrheniusRate:
            A = rxn.rate.pre_exponential_factor * UF[i] ** normA[i]
            rxn.rate = ct.ArrheniusRate(A,
                                        rxn.rate.temperature_exponent,
                                        rxn.rate.activation_energy)
        elif type(rxn.rate) == ct.PlogRate:
            A = rxn.rates[1][1].pre_exponential_factor * UF[i] ** normA[i]
            rxn.rates[1] = (1 * ct.one_atm, ct.ArrheniusRate(A,
                                                             rxn.rates[1][1].temperature_exponent,
                                                             rxn.rates[1][1].activation_energy))
        else:
            A_low = rxn.rate.low_rate.pre_exponential_factor * UF[i] ** normA[i]
            i += 1
            A_high = rxn.rate.high_rate.pre_exponential_factor * UF[i] ** normA[i]
            if type(rxn.rate) == ct.LindemannRate:
                rxn.rate = ct.LindemannRate(low=ct.Arrhenius(A_low,
                                                             rxn.rate.low_rate.temperature_exponent,
                                                             rxn.rate.low_rate.activation_energy),
                                            high=ct.Arrhenius(A_high,
                                                              rxn.rate.high_rate.temperature_exponent,
                                                              rxn.rate.high_rate.activation_energy),
                                            falloff_coeffs=rxn.rate.falloff_coeffs)
            else:
                rxn.rate = ct.TroeRate(low=ct.Arrhenius(A_low,
                                                        rxn.rate.low_rate.temperature_exponent,
                                                        rxn.rate.low_rate.activation_energy),
                                       high=ct.Arrhenius(A_high,
                                                         rxn.rate.high_rate.temperature_exponent,
                                                         rxn.rate.high_rate.activation_energy),
                                       falloff_coeffs=rxn.rate.falloff_coeffs)
        gas.modify_reaction(idx, rxn)
        i += 1

    return gas


def get_A(gas, UF):
    A = np.zeros_like(UF)
    i = 0
    reactions = gas.reactions()
    for rxn in reactions:
        if isinstance(rxn.rate, ct.ArrheniusRate):
            A[i] = rxn.rate.pre_exponential_factor
        elif isinstance(rxn.rate, ct.PlogRate):
            A[i] = rxn.rates[1][1].pre_exponential_factor
        else:
            A[i] = rxn.rate.low_rate.pre_exponential_factor
            i += 1
            A[i] = rxn.rate.high_rate.pre_exponential_factor
        i += 1

    return A


def calc_st_idt(condition):
    mech_path = condition['mech_path']
    normA = condition['normA']
    UF = condition['UF']
    T, P = condition['TP'][0], condition['TP'][1]
    phi = condition['phi']
    XFO = condition['XFO']
    exp_idt = condition['idt']

    gas = ct.Solution(mech_path)
    gas = modify_A(gas, normA, UF)
    gas.TP = T, ct.one_atm * P
    gas.set_equivalence_ratio(phi, fuel=XFO['fuel'], oxidizer=XFO['oxidizer'])
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])

    time = []
    temp = []

    ctrl_dt = 1e-6
    n_steps = 10 * int(exp_idt * 1e-3 / ctrl_dt)
    try:
        for _ in range(n_steps):
            sim.advance(sim.time + ctrl_dt)
            time.append(sim.time + ctrl_dt)
            temp.append(gas.T)
        diff_temp = np.diff(temp) / np.diff(time)
        ign = time[np.argmax(diff_temp)]
    except Exception as e:
        ign = 1e-8
    return ign


def calc_fitness(idt, exp_idt, group_indices):
    rel_err = ((np.log(idt * 1000 / exp_idt)) / 0.1 / np.log(exp_idt)) ** 2
    fitness = 0
    for i in range(len(group_indices) - 1):
        start_idx = group_indices[i]
        end_idx = group_indices[i + 1]
        fitness += np.mean(rel_err[:, start_idx:end_idx], axis=1)

    fitness /= len(group_indices) - 1

    return fitness