# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:30:18 2019

@author: JL7616
"""

import numpy as np
import matplotlib.pyplot as plt
from network import PRA

class Ensemble():
    
    def __init__(self,N_list,m_list):
        self._N_list = N_list
        self._m_list = m_list
        self._ensemble_list = self.initialisation()
        
    def initialisation(self):
        self._l = []
        for i in self._N_list:
            for j in self._m_list:
                self._l.append(PRA(i,j))
        return self._l
    
    def simulation(self):
        for i in range(len(self._l)):
            self._ensemble_list[i].simulation()
            print ("N=", self._ensemble_list[i]._N, "m=", self._ensemble_list[i]._m, "completed")
            
    def total_realisations(self,realisation):
        for i in range(len(self._l)):
            self._ensemble_list[i].total_realisations(realisation)
            print ("N=", self._ensemble_list[i]._N, "m=", self._ensemble_list[i]._m, "completed")
            
"""fixed m, varying N"""
N_list = [1e2,1e3,1e4,1e5,1e6]  
m_list = [3]  
networks = Ensemble(N_list,m_list)
networks.total_realisations(100)

scale = 1.20

# linear fit for log scale
def fit_log(list0, list1):
    loglist0 = [np.log(i) for i in list0] 
    loglist1 = [np.log(i) for i in list1] 
    fit = np.polyfit(loglist0,loglist1,1)
    #print ('A=',np.exp(fit[0[0]]),'n=',fit[0][1])
    print ('value=',fit)
    fit_fn = np.poly1d(fit)
    fit_output = [np.exp(fit_fn[0])*(i**fit_fn[1]) for i in list0]
    return fit_output

# calculate average k1, standard deviation of average k1, theoretical k1
avg_k1_list = []
k1_std_list = []
theoretical_k1_list=[] 
for i in range(len(N_list)):
    k1_list_N = networks._ensemble_list[i].k1_list()
    avg_k1_list.append(np.mean(k1_list_N))
    k1_std_list.append(np.std(k1_list_N)/np.sqrt(100))
    theoretical_k1_list.append(networks._ensemble_list[i].theoretical_k1())

f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
f.tight_layout() 
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.scatter(N_list,avg_k1_list,label = 'experimental data',s=15)
ax1.errorbar(N_list,avg_k1_list,yerr=k1_std_list,label = 'standard deviation', fmt = 'none')
#plt.plot(N_list,fit_log(N_list,avg_k1_list),linestyle='--',label = r'linear fit $e^{1.135}N^{0.500}$')
ax1.plot(N_list,theoretical_k1_list,label = 'theoretical prediction')
ax1.legend(prop={'size': 10})
ax1.set_xlabel("N",fontsize = 12.5)
ax1.set_ylabel(r"$\langle k_1 \rangle$",fontsize = 12.5)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.scatter(N_list,k1_std_list,label = 'experimental data',s=15)
#plt.plot(N_list,fit_log(N_list,k1_std_list),linestyle='--',label = r'linear fit $e^{-3.022}N^{0.515}$')
ax2.set_xlabel("N",fontsize = 12.5)
ax2.set_ylabel(r"$\sigma_{\langle k_1 \rangle}$",fontsize = 12.5)
ax2.legend(prop={'size': 10})

# plot log-binned pk()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
f.tight_layout() 
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel("p(k)",fontsize = 12.5)
ax1.set_xlabel("k",fontsize = 12.5)
for i in range(len(N_list)):
    k_value, k_prob = networks._ensemble_list[i].logbin_total_k(scale)
    ax1.plot(k_value, k_prob)
ax1.legend([r'$m=10^2$',r'$m=10^3$',r'$m=10^4$',r'$m=10^5$',r'$m=10^6$'],loc=3,prop={'size': 10})

# plot theoretical p(k) (the long time limit p)
k_value, k_prob = networks._ensemble_list[-1].logbin_total_k(scale)
theoretical_k_prob = networks._ensemble_list[-1].theoretical_k_prob_list()
ax1.plot(k_value, theoretical_k_prob, 'grey')
   
# plot collapsed log-binned k data
fig = plt.figure()
ax2.set_xscale('log')
ax2.set_yscale('log')
for i in range(len(N_list)):
    k_value, k_prob = networks._ensemble_list[i].logbin_total_k(scale)
    scaled_k_value = k_value/theoretical_k1_list[i]
    theoretical_k_prob = networks._ensemble_list[i].theoretical_k_prob_list()
    scaled_k_prob = k_prob/np.asarray(theoretical_k_prob)
    ax2.plot(scaled_k_value,scaled_k_prob,'-o',markersize=3)
ax2.set_xlabel(r"$k/k_1$",fontsize = 12.5)
ax2.set_ylabel(r"$p(k)/p_{\infty}(k)$",fontsize = 12.5)
ax2.legend([r'$N=10^2$',r'$N=10^3$',r'$N=10^4$',r'$N=10^5$',r'$N=10^6$'],loc=3,prop={'size': 10})

 