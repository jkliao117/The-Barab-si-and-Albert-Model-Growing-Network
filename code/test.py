#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:07:50 2019

@author: RoroLiao
"""
import numpy as np
import scipy.stats as sps
import networkx as nx
import matplotlib.pyplot as plt
import random 
from logbin import logbin
                
class BA_model():
    
    def __init__(self,N,m):
        self._N = N
        self._m = m
        self._G = nx.complete_graph(m+1)    

    def simulation(self):
        self._E_list = []
        self._avg_k_list = []
        self._edge_node_list = list(np.hstack(self._G.edges()))
        while self._G.number_of_nodes() < self._N:
            new_node = len(self._G)         
            connected_nodes = self.connected_nodes(new_node)
            self._G.add_node(new_node)
            self._G.add_edges_from([(new_node,i) for i in connected_nodes])
            # calculate the total number of edges at a time step
            self._E_list.append(self._G.number_of_edges())
            # calculate the average degree at a time step
            self._avg_k_list.append(np.mean([self._G.degree(i) for i in range(int(self._G.number_of_nodes()))]))
    
    def reset(self):
        self._G = nx.complete_graph(self._m+1)  

    def k(self):
        return self._k_value, self._k_prob
        
    def k_list(self):
        return self._k_list
    
    def logbin_k(self,scale):
        self._lb_k_value, self._lb_k_prob = logbin(self.k_list(),scale)
        return self._lb_k_value, self._lb_k_prob
    
    def E(self):
        return self._E_list
    
    def avg_k(self):
        return self._avg_k_list
    

class PPA(BA_model):
    
    def connected_nodes(self,new_node):
        nodes_to_connect = []
        while len(nodes_to_connect) < self._m:
            K = len(self._edge_node_list)
            index = random.randint(0,K-1)
            node = self._edge_node_list[index]
            if (node in nodes_to_connect) == False:
                nodes_to_connect.append(node)
        self._edge_node_list.extend(nodes_to_connect)
        self._edge_node_list.extend(self._m*[new_node])
        return nodes_to_connect
               
class PRA(BA_model):
    
    def connected_nodes(self,new_node):
        nodes_to_connect = random.sample(list(self._G.nodes()), self._m)
        return nodes_to_connect
 
"""test the program"""    
# plot the time evolution of the total number of edges andthe average degree
m_test_list = [3,9,27]
EmN_array = np.zeros((len(m_test_list),int(2e3)))
k2m_array = np.zeros((len(m_test_list),int(2e3)))
for i in range(len(m_test_list)):
    m = m_test_list[i]
    N = 2e3+(m+1)
    network_test = PPA(N,m)
    network_test.simulation()
    E = network_test.E()
    avg_k = network_test.avg_k()
    N_array = np.arange(m+1,N)
    EmN_array[i] = np.asarray(E)/(m*N_array)
    k2m_array[i] = np.asarray(avg_k)/(2*m)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
for i in range(len(m_test_list)):
    ax1.plot(np.arange(2e3),EmN_array[i])
ax1.hlines(1,0,2e3,linestyles='dashed')
ax1.xlabel('time t')
ax1.ylabel('E(t)/mN(t)')
ax1.legend(['m=3','m=9','m=27'])
for i in range(len(m_test_list)):
    ax2.plot(np.arange(2e3),k2m_array[i])
ax2.hlines(1,0,2e3,linestyles='dashed')
ax2.xlabel('time t')
ax2.ylabel(r'$\langle k\rangle/2m$')
ax2.legend(['m=3','m=9','m=27'])