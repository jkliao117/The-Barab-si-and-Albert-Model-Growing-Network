#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 06:50:47 2019

@author: RoroLiao
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from network import EVM

"""fixed N, varying m"""

class Ensemble():
    
    def __init__(self,N_list,m_list):
        self._N_list = N_list
        self._m_list = m_list
        self._ensemble_list = self.initialisation()
        
    def initialisation(self):
        self._l = []
        for i in self._N_list:
            for j in self._m_list:
                self._l.append(EVM(i,j))
        return self._l
    
    def simulation(self):
        for i in range(len(self._l)):
            self._ensemble_list[i].simulation()
            print ("N=", self._ensemble_list[i]._N, "m=", self._ensemble_list[i]._m, "completed")
            
    def total_realisations(self,realisation):
        for i in range(len(self._l)):
            self._ensemble_list[i].total_realisations(realisation)
            print ("N=", self._ensemble_list[i]._N, "m=", self._ensemble_list[i]._m, "completed")
    

"""fixed N, varying m"""
N_list = [1e3]    
m_list = [3,9,27]  
networks = Ensemble(N_list,m_list)
networks.total_realisations(100)

scale = 1.15

color =  ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
fig = plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.ylabel("p(k)")
plt.xlabel("k")

chisq_list = []
p_list = []
df_list = []
# plot raw p(k)
for i in range(len(m_list)):
    k_value1, k_prob1 = networks._ensemble_list[i].logbin_total_k(1)
    plt.scatter(k_value1, k_prob1, s=10, color=color[i],alpha=0.5)
# plot log-binned p(k)
for i in range(len(m_list)):
    k_value, k_prob = networks._ensemble_list[i].logbin_total_k(scale)
    plt.plot(k_value, k_prob, color=color[i])   
# plot theoretical p(k) and perform chi squared test
chisq_list = []
p_list = []
df_list = []
for i in range(len(m_list)):
    k_value, k_prob = networks._ensemble_list[i].logbin_total_k(scale)
    theoretical_k_prob = networks._ensemble_list[i].theoretical_k_prob_list()
    plt.plot(k_value, theoretical_k_prob, color=color[i],linestyle = '--')
    chisq, p = sps.chisquare(k_prob,theoretical_k_prob)
    chisq_list.append(chisq)
    p_list.append(p)
    df_list.append(len(k_prob)-1)
    
plt.legend(['m=3','m=9','m=27'],loc=3)
