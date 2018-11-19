#!/usr/bin/python
# Author: Suzanna Sia

import sys
import vtagem
import pdb

numIterations = 10

hmm = vtagem.HMM(sys.argv[1], sys.argv[3], sys.argv[2])
hmm.train(sys.argv[1]) # Initialisation of model parameters
hmm.collect_raw_statistics()
hmm.collect_test_statistics()
hmm.initialise_on_train()
hmm.test_and_eval()
#pdb.set_trace()

for itr in range(numIterations):
    print("Iteration "+str(itr)+":",end="\t")
    hmm.doEStep() # compute new counts on raw
    hmm.doMStep() # resetimate model parameters
    hmm.test_and_eval() # recode test data



#hmm.test_and_eval(sys.argv[2])
