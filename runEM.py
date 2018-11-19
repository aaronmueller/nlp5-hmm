#!/usr/bin/python
# Author: Suzanna Sia

import sys
import vtagem
import pdb

numIterations = 10

hmm = vtagem.HMM(sys.argv[1], sys.argv[2], sys.argv[3])
hmm.train(sys.argv[1]) # Initialisation of model parameters
hmm.collect_test_statistics()
hmm.collect_raw_statistics()
hmm.initialise_on_train()
hmm.test_and_eval()
#pdb.set_trace()

for itr in range(numIterations):
    print("Iteration:"+str(itr))
    print("Estep..")
    hmm.doEStep() # compute new counts on raw
    print("Mstep..")
    hmm.doMStep() # resetimate model parameters
    print("Test eval..")
    hmm.test_and_eval() # recode test data



#hmm.test_and_eval(sys.argv[2])
