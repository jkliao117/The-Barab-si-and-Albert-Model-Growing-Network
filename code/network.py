#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:23:57 2019

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
        # number of nodes
        self._N = N
        # number of edges added for each new nodes 
        self._m = m
        # initialise graph (first instance)
        # a complete graph with m+1 nodes
        self._G = nx.complete_graph(m+1)    

    def simulation(self):
        # create the attachement list
        self._edge_node_list = list(np.hstack(self._G.edges()))
        self._node_list = list(self._G.nodes())
        # before there are N nodes in the network
        while self._G.number_of_nodes() < self._N:
            # set the index for the new node
            new_node = len(self._G)
            # find which nodes to connect            
            connected_nodes = self.connected_nodes(new_node)
            # add the new node to the graph
            self._G.add_node(new_node)
            # add m edges according to the probability 
            self._G.add_edges_from([(new_node,i) for i in connected_nodes])
            self._node_list.append(new_node)
        # find the probability distribution of degree k from histogram and remove k = 0
        degree_list = nx.degree_histogram(self._G)
        # remove k = 0
        self._k_value = np.delete(np.arange(len(degree_list)),[0])
        # normalised probability
        self._k_prob =  np.delete(np.array(degree_list)/len(self._G.nodes()),[0])
        # list of k
        self._k_list = [self._G.degree(i) for i in range(int(self._N))]
    
    def reset(self):
        self._G = nx.complete_graph(self._m+1)  

    # find values of k and the corresponding probability
    def k(self):
        return self._k_value, self._k_prob
        
    # find the degree k for all nodes
    def k_list(self):
        return self._k_list
    
    # log-binned k data
    def logbin_k(self,scale):
        self._lb_k_value, self._lb_k_prob = logbin(self.k_list(),scale)
        return self._lb_k_value, self._lb_k_prob
      
    # repeat simulation to create a total k list
    def total_realisations(self,realisation):
        i=0
        self._total_k_list = []
        self._k1_list = []
        # before iterating over realisations
        while i < int(realisation):
            # run simulation
            self.reset()
            self.simulation()
            k_list = self.k_list()
            self._total_k_list.extend(k_list)
            self._k1_list.append(max(k_list))
            i += 1
            print(i)
    
    # log bin the total k list
    def logbin_total_k(self,scale):
        self._lb_total_k_value, self._lb_total_k_prob = logbin(self._total_k_list,scale)
        self._lb_total_k_value = np.delete(self._lb_total_k_value,0)
        self._lb_total_k_prob = np.delete(self._lb_total_k_prob,0)
        return self._lb_total_k_value, self._lb_total_k_prob
        
    # find the list for all k1
    def k1_list(self):
        return self._k1_list
     
    # visualise the network
    def plot_network(self):
        nx.draw(self._G)
    
    # evaluate chisq and p values           
    def Chi_Squared_test(self):
        theoretical_k_prob_list = self.theoretical_k_prob_list()
        chisq, p = sps.chisquare(self._lb_k_prob,theoretical_k_prob_list)
        print ("Chi Squared = " , chisq, "p value =", p)
        

class PPA(BA_model):
    
    def connected_nodes(self,new_node):
        nodes_to_connect = []
        # before there are m edges in accepted
        while len(nodes_to_connect) < self._m:
            # select p(k) propto k
            # uniformly sample by index
            K = len(self._edge_node_list)
            index = random.randint(0,K-1)
            node = self._edge_node_list[index]
            # avoid repeats
            if (node in nodes_to_connect) == False:
                nodes_to_connect.append(node)
        # update the attachment list
        self._edge_node_list.extend(nodes_to_connect)
        self._edge_node_list.extend(self._m*[new_node])
        return nodes_to_connect
                
    def theoretical_k_prob_list(self):
        theoretical_k_prob_list = []
        m = self._m
        for k in self._lb_total_k_value:
            theoretical_k_prob_list.append((2*m*(m+1))/(k*(k+1)*(k+2)))
        return theoretical_k_prob_list
    
    def theoretical_k1(self):
        N = self._N
        m = self._m
        return np.sqrt(0.25 + N*m*(m+1)) - 0.5

           
class PRA(BA_model):
    
    def connected_nodes(self,new_node):
        # randomly sample nodes
        nodes_to_connect = random.sample(self._node_list, self._m)
        return nodes_to_connect
        
    def theoretical_k_prob_list(self):
        theoretical_k_prob_list = []
        m = self._m
        for k in self._lb_total_k_value:
            theoretical_k_prob_list.append(((m/(m+1))**(k-m))/(m+1))
        return theoretical_k_prob_list
    
    def theoretical_k1(self):
        N = self._N
        m = self._m
        return m -(np.log(N)/(np.log(m)-np.log(m+1)))
        
class EVM(BA_model):
    
    def simulation(self):
        self._edge_node_list = list(np.hstack(self._G.edges()))
        self._node_list = list(self._G.nodes())
        self._edge_list = list(self._G.edges())
        while self._G.number_of_nodes() < self._N:
            new_node = len(self._G)           
            #PPA
            connected_nodes = self.connected_nodes_1(new_node)
            #PRA
            nodes_pairs = self.connected_nodes_2()
            self._G.add_node(new_node)
            self._node_list.append(new_node)
            #PPA
            self._G.add_edges_from([(new_node,i) for i in connected_nodes])
            #PRA
            self._G.add_edges_from([i for i in nodes_pairs])
        degree_list = nx.degree_histogram(self._G)
        self._k_value = np.delete(np.arange(len(degree_list)),[0])
        self._k_prob =  np.delete(np.array(degree_list)/len(self._G.nodes()),[0])
        self._k_list = [self._G.degree(i) for i in range(int(self._N))]
      
    #PPA
    def connected_nodes_1(self, new_node):
        r = int(np.ceil(self._m/2))
        nodes_to_connect = []
        while len(nodes_to_connect) < r:
            K = len(self._edge_node_list)
            index = random.randint(0,K-1)
            node = self._edge_node_list[index]
            if (node in nodes_to_connect) == False:
                nodes_to_connect.append(node)
        self._edge_node_list.extend(nodes_to_connect)
        self._edge_node_list.extend(self._m*[new_node])
        for j in nodes_to_connect:
            self._edge_list.append(tuple([new_node,j]))
        return nodes_to_connect
    
    #PRA           
    def connected_nodes_2(self):
        r = int(np.floor(self._m/2))
        nodes_pairs = []
        while len(nodes_pairs) < r:
            pair = tuple(random.sample(self._node_list, 2))
            if ((pair in nodes_pairs)== False) and ((pair in self._edge_list)== False):
                nodes_pairs.append(pair)
                self._edge_node_list.extend(list(pair))
        self._edge_list.extend(nodes_pairs)
        return nodes_pairs
    
    def theoretical_k_prob_list(self):
        theoretical_k_prob_list = []
        m = self._m
        for k in self._lb_total_k_value:
            a = k+4*m
            b = 9*m
            A = b*(b+2)*(b+4)*(b+6)/4
            B = a*(a+1)*(a+2)*(a+3)*(a+4)
            theoretical_k_prob_list.append(A/B)
        return theoretical_k_prob_list
    
    