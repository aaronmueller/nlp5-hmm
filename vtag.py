'''
Name: Aaron Mueller & Suzanna Sia
Course: EN.601.665 -- Natural Language Processing
Instructor: Jason Eisner
Date: 15 Nov. 2018
Assignment: HW 6 -- HMMs
'''

import sys

class HMM:
    def __init__(self, states, observations, start_probs, transition_probs,
            emission_probs):
        self.states = states
        self.observations = observations
        self.start_probs = start_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs

# states for ice cream example
hmm = HMM(('H', 'C'), ('1', '2', '3'), {'H': 0.5, 'C': 0.5},
        {'H': {'H': 0.8, 'C': 0.1}, 'C': {'H': 0.1, 'C': 0.8}, '###': {'H': 0.1, 'C': 0.1}},
        {'1': {'H': 0.1, 'C': 0.7}, '2': {'H': 0.2, 'C': 0.2}, '3': {'H': 0.7, 'C': 0.1}})

### VITERBI
# Based on the algorithm on page R-4 in handout
in_file = sys.argv[1]
mu_t = [1]
prev_word = 'START'
back_t = []

true_tags = []

# TODO: handle start and end states
# also handle multiple sentences (now, only handles first sentence)
with open(in_file, 'r') as f:
    # TODO: handle SOS
    f.readline()

    for i, line in enumerate(f):
        w_i, tag = line.strip().split('/')
        true_tags.append(tag)
        
        # TODO: handle EOS
        if w_i == '###':
            break

        tags = hmm.emission_probs[w_i].keys()
        if prev_word == 'START':
            prev_tags = hmm.start_probs.keys()
        else:
            prev_tags = hmm.emission_probs[prev_word].keys()
        
        # initialize mu_t[i] and back_t[i]
        mu_t.append(0)
        back_t.append('')

        for t_i in tags:
            for t_im1 in prev_tags:
                p = hmm.transition_probs[t_i][t_im1] * hmm.emission_probs[w_i][t_i]
                mu = mu_t[i-1] * p
                if mu > mu_t[i]:
                    mu_t[i] = mu
                    back_t[i] = t_im1

   
    # go through backpointers, recover most likely tags
    pred_tags = ['###']
    for i in range(len(back_t)-1, 1, -1):
        pred_tags.append(back_t[i])

    # reverse array to get sequence of tags in sentence order
    pred_tags = pred_tags[::-1]
    
    # compare
    # TODO: implement accuracy metrics (and time?)
    print(pred_tags)
    print(true_tags)
