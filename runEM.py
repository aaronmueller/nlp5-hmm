#!/usr/bin/python
# Author: Suzanna Sia

import sys
import vtagem
import pdb

numIterations = 1

hmm = vtagem.HMM(sys.argv[1], sys.argv[2], sys.argv[3])
hmm.train(sys.argv[1]) # Initialisation of model parameters
#hmm.prepare_for_em()
hmm.doMStep()
hmm.test_and_eval()
#pdb.set_trace()

#for itr in range(numIterations):
#    hmm.doEStep() # compute new counts on raw
#    hmm.doMStep() # resetimate model parameters
#    hmm.test_and_eval() # recode test data



#hmm.test_and_eval(sys.argv[2])
