#!/usr/bin/python
# Author: Suzanna Sia

import sys
import vtagem_cz
import pdb
import time

numIterations = 10

hmm = vtagem_cz.HMM(sys.argv[1], sys.argv[3], sys.argv[2])
hmm.train(sys.argv[1]) # Initialisation of model parameters
hmm.collect_raw_statistics()
hmm.collect_test_statistics()
hmm.initialise_on_train()
hmm.test_and_eval()
#pdb.set_trace()

for itr in range(numIterations+1):
    start = time.time()
    print("Iteration "+str(itr)+":",end="\t")
    hmm.doEStep() # compute new counts on raw
    hmm.doMStep() # resetimate model parameters
    hmm.test_and_eval() # recode test data

    print("# Time Elapsed:", time.time() - start)



#hmm.test_and_eval(sys.argv[2])
