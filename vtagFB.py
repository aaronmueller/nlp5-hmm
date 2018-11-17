'''
Name: Aaron Mueller & Suzanna Sia
Course: EN.601.665 -- Natural Language Processing
Instructor: Jason Eisner
Date: 15 Nov. 2018
Assignment: HW 6 -- HMMs
'''

import sys
import numpy as np
import pdb
from collections import defaultdict
import math

class HMM:
    def __init__(self):
        self.states = set()
        self.observations = set()
        self.start_probs = {}
        self.transition_probs = {}
        self.emission_probs = {}
        self.tag_dict = {}
        self.alpha_t = {}
        self.beta_t = {}
        self.mu_t = {}
        self.back_t = {}

    # fill variables declared in __init__
    def train(self, train_file):
        with open(train_file, 'r') as f:
            start_counts = defaultdict(int)
            num_sentences = 0
            num_words = 0
            transition_counts = {}
            emission_counts = {}
            
            SOS = False

            lamda = 1
            self.observations.add('<OOV>')
            self.tag_dict['<OOV>'] = set()

            for line_no, line in enumerate(f):
                word, tag = line.strip().split('/')

                if word not in self.observations:
                    self.observations.add(word)
                if tag not in self.states:
                    self.states.add(tag)

                if word == "###":
                    SOS = True
                    # handle EOS
                    if line_no != 0:
                        if prev_tag not in transition_counts:
                            transition_counts[prev_tag] = defaultdict(int)
                        transition_counts[prev_tag][tag] += 1
                        
                        if tag not in emission_counts:
                            emission_counts[tag] = defaultdict(int)
                        emission_counts[tag][word] += 1

                    prev_tag = "###"
                    continue
                    
                if SOS:
                    start_counts[tag] += 1
                    num_sentences += 1
                    SOS = False
                

                # Unsmoothed version. If tag has not been seen in training set will assign
                # 0 prob
                if word not in self.tag_dict.keys():
                    # self.tag_dict[word] = ['H', 'C']
                    self.tag_dict[word] = set()
                if tag not in self.tag_dict[word]:
                    self.tag_dict[word].add(tag)
                
                if prev_tag not in transition_counts:
                    transition_counts[prev_tag] = defaultdict(int)
                transition_counts[prev_tag][tag] += 1

                if tag not in emission_counts:
                    emission_counts[tag] = defaultdict(int)
                emission_counts[tag][word] += 1

                num_words += 1
                prev_tag = tag


            for tag in start_counts.keys():
                self.start_probs[tag] = float(start_counts[tag]) / num_sentences

            for prev_tag in transition_counts:
                self.transition_probs[prev_tag] = {}
                transition_sum = sum(transition_counts[prev_tag].values())
                
                # add-lambda smoothing
                for tag in self.states:

                    if tag in transition_counts[prev_tag]:
                        self.transition_probs[prev_tag][tag] = \
                                float(transition_counts[prev_tag][tag] + lamda) / \
                                (transition_sum + (lamda * len(self.states)))
                    else:
                        self.transition_probs[prev_tag][tag] = \
                                lamda / (transition_sum + lamda * len(self.states))
            
            
            #for word in emission_counts:
            #    self.emission_probs[word] = {}
            #    emission_sum = sum(emission_counts[word].values())
            #    for tag in emission_counts[word]:
            #        self.emission_probs[word][tag] = \
            #                float(emission_counts[word][tag]) / emission_sum


            for tag in emission_counts:
                self.emission_probs[tag] = {}
                emission_sum = sum(emission_counts[tag].values())

                # add-lambda smoothing
                for word in self.observations:
                    if word in emission_counts[tag]:
                        self.emission_probs[tag][word] = \
                                float(emission_counts[tag][word] + lamda) / \
                                (emission_sum + (lamda * len(self.observations)))
                    else:
                        self.emission_probs[tag][word] = \
                                lamda / (emission_sum + (lamda * len(self.observations)))

        self.tag_dict['<OOV>'].update(self.states)
        self.tag_dict['<OOV>'].difference_update({'###'})
        # print("transition probabilities:", self.transition_probs)

        # print("emission probabilities:", self.emission_probs)
        print()
        

    def calc_state_probs(self, i, w_i, prev_word, direction=None):
        # if forward, state_probs = self.alpha_t
        # if backward, state_probs = self.beta_t

        if direction =="forward":
            state_p = self.alpha_t
        
        elif direction == "backward":
            state_p = self.beta_t

        if w_i == "###":
            # SOS
            if prev_word == '':
                prev_word = "###"
                return prev_word
            # EOS
            else:
                tags = {'###'}
                prev_tags = self.tag_dict[prev_word]
        
        # First word
        elif prev_word == '###':
            tags = self.tag_dict[w_i]
            prev_tags = {'###'} # ### tags ###

        else:
            tags = self.tag_dict[w_i]
            prev_tags = self.tag_dict[prev_word]
        
        
        for t_i in tags:
            for t_im1 in prev_tags:
                p = math.log(self.transition_probs[t_im1][t_i]) + \
                    math.log(self.emission_probs[t_i][w_i])
                
                mu = state_p[t_im1][i-1] + p
                # Cannot just add log probs here. 
                # we need to do log sum exponentials as explained on R-10
                state_p[t_i][i] = np.logaddexp(state_p[t_i][i], mu)


    def get_max_tag(self, i, w_i):

        if w_i=="###":
            return "###"

        max_tag_score = float('-inf')
        max_tag = None

        for tag in self.tag_dict[w_i]:
            predicted_tag_score = self.alpha_t[tag][-i-1] + self.beta_t[tag][i]
            
            # Line 13 of pseudocode normalises but there is no need to normalise since we are
            # taking the max.

            if predicted_tag_score > max_tag_score:
                max_tag = tag
                max_tag_score = predicted_tag_score

        return max_tag


    def test_and_eval(self, test_file):
        ### VITERBI
        # Based on the algorithm on page R-4 in handout
        self.mu_t = {}
        prev_word = ''
        self.back_t = {}

        self.alpha_t = {}
        self.beta_t = {}
        #self.back_t = ['']
        # Similar to back_t in viterbi
        predicted_tags = []
        true_tags = []
        words = []

        crossentropy = 0

        with open(test_file, 'r') as f:
            testlines = f.readlines()
        
        known = [True] * len(testlines)
       
        # Initialising alpha_t and beta_t 
        for state in self.states:

            self.alpha_t[state] = [math.log(1)]
            self.beta_t[state] = [math.log(1)]

            for i in range(1, len(testlines)):
                self.alpha_t[state].append(float("-inf"))
                self.beta_t[state].append(float("-inf"))

        # forward pass

        for i in range(len(testlines)):
            line = testlines[i]
            w_i, tag = line.strip().split('/')

            words.append(w_i)
            true_tags.append(tag)

            if w_i not in self.observations:
                w_i = '<OOV>'
                known[i] = False

            if i > 0:
                crossentropy += math.log(self.transition_probs[true_tags[i-1]][true_tags[i]]) \
                        + math.log(self.emission_probs[tag][w_i])
            self.calc_state_probs(i, w_i, prev_word, direction="forward")
            prev_word = w_i
       
        # backward pass
        #   note beta_t is reversed indexed in order to reuse the way initialisation works 
        #   i.e., beta_t[0] is the state of the last observed word.
        #   hence predicted_tags is reversed indexed as well.

        prev_word = ''
        self.mu_t = {}
        self.back_t = {}
        for i in range(len(testlines)):

            line = testlines[len(testlines)-1-i]
            w_i, tag = line.strip().split('/')

            if w_i not in self.observations:
                w_i = '<OOV>'

            self.calc_state_probs(i, w_i, prev_word, direction="backward")
            prev_word = w_i
            
            predicted_tags.append(self.get_max_tag(i, w_i))
        # Sanity check
        #for i in range(len(self.beta_t['H'])):
        #    print(i, math.exp(self.beta_t['H'][i]))
        
        # compare and evaluate
        correct = 0
        total = 0
        known_total = 0
        known_correct = 0
        unknown_correct = 0
        tags = predicted_tags[::-1]

        for i in range(0, len(true_tags)):
            if true_tags[i] == tags[i] and true_tags[i] != "###":
                correct += 1
                total += 1
                if known[i] == True:
                    known_total += 1
                    known_correct += 1
                else:
                    unknown_correct += 1
            elif true_tags[i] != "###":
                total += 1
                if known[i] == True:
                    known_total += 1
        
        unknown_total = total - known_total
        print("accuracy: {}".format(float(correct) / total))
        print("known accuracy: {}".format(float(known_correct) / known_total))
        if unknown_total > 0:
            print("unknown accuracy: {}".format(float(unknown_correct) / unknown_total))

        #print("predicted tags: {}\nlength: {}".format(tags, len(tags)))
        #for i in range(10):
        #    print(tags[-i], end=" ")
        #    print(true_tags[-i])
        #print("true tags: {}\nlength: {}".format(true_tags, len(true_tags)))     

        # Calculate perplexity at the end of forwardpass
        # length -1 because of ###/### at SOS
        # 
        # self.alpha_t['###'] should contain the sum of alpha_t
        perplexity = math.exp( - crossentropy / (len(testlines)-1))
        print("perplexity:", perplexity)


        output = []
        for i in range(len(test_file)):
            output.append("{}\{}".format(words[i], predicted_tags[i]))

        with open('test-output', 'w') as f:
            output.append("###\###")
            f.write("\n".join(output))


# run Hidden Markov Model on specified training and testing files
hmm = HMM()
hmm.train(sys.argv[1])
hmm.test_and_eval(sys.argv[2])
