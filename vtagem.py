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
from collections import Counter
import math

class HMM:
    def __init__(self, train_file, raw_file, test_file):

        self.lamda = 1

        self.states = set()
        self.observations = set()

        # original counts
        self.org_transition_counts = {}
        self.org_emission_counts = {}
        self.org_state_counts = defaultdict(int)
        self.org_tag_dict = {}
        self.org_observations = set()


        # previous counts
        # current counts
        self.tag_dict = {}
        self.transition_probs = {}
        self.emission_probs = {}

        self.alpha_t = {}
        self.beta_t = {}
        self.mu_t = {}
        self.back_t = {}

        with open(train_file, 'r') as f:
            self.trainlines = f.readlines()

        with open(raw_file, 'r') as f:
            self.rawlines = f.readlines()

        with open(test_file, 'r') as f:
            self.testlines = f.readlines()


    # fill variables declared in __init__

    def get_suffix(self, word):
        #
        # uncomment "return word" below to remove the effect of suffix
        #
        return word
        # -4 is the best length so far
        # from 92.65% to 93.59%

        if len(word)>3:
            sfx = word[-4:]
        else:
            sfx = word
        return sfx
    
    def add_count(self, count, ind1, ind2, count_dict):
        if ind1 not in count_dict:
            count_dict[ind1] = defaultdict(int)
        count_dict[ind1][ind2] +=count
        return count_dict

    def calc_smooth_probs(self, count_dict, probs, vals):

        #count_dict = transition_counts
        #probs = transition_probs
        #vals = self.states or self.observations
        for index1 in count_dict:
            probs[index1] = {}
            normalising_sum = sum(count_dict[index1].values())

            for index2 in vals:
                if index2 in count_dict[index1]:
                    probs[index1][index2] = \
                        float(count_dict[index1][index2] + self.lamda)/ \
                        (normalising_sum + (self.lamda * len(vals)))
                else:
                    probs[index1][index2] = \
                        self.lamda/ (normalising_sum + self.lamda * len(vals))

    def train(self, train_file):
        # Initialises counts for parameter estimation
        # Note we need to keep original counts

        start_counts = defaultdict(int)
        
        num_words = 0
        self.org_state_counts = defaultdict(int)
        
        SOS = False

        #lamda = 1
        for line_no, line in enumerate(self.trainlines):
            word, tag = line.strip().split('/')
            
            #sfx = self.get_suffix(word)

            if word not in self.org_observations:
                self.org_observations.add(word)
            #    self.sfx_org_observations.add(sfx)

            if tag not in self.states:
                self.states.add(tag)

            if word == "###":
                SOS = True
                # handle EOS
                if line_no != 0:
                    
                    self.org_transition_counts = self.add_count(1, prev_tag, tag, self.org_transition_counts)
                    self.org_emission_counts = self.add_count(1, tag,word, self.org_emission_counts)
                    #self.sfx_org_emission_counts = self.add_count(1, tag, sfx,
                    #        self.sfx_org_emission_counts)

                prev_tag = "###"
                continue
                
            if SOS:
                start_counts[tag] += 1
                SOS = False
           
            self.org_state_counts[tag] += 1

            if word not in self.org_tag_dict.keys():
                self.org_tag_dict[word] = set()

            self.org_tag_dict[word].add(tag)

            #if sfx not in self.sfx_org_tag_dict.keys():
            #    self.sfx_org_tag_dict[sfx] = set()
            #self.sfx_org_tag_dict[sfx].add(tag)

            self.org_transition_counts = self.add_count(1, prev_tag, tag, self.org_transition_counts)             
            self.org_emission_counts = self.add_count(1, tag, word, self.org_emission_counts)
            #self.sfx_org_emission_counts = self.add_count(1, tag, sfx, self.sfx_org_emission_counts)

            num_words += 1
            prev_tag = tag

        self.org_observations.add('<OOV>')
        self.org_tag_dict['<OOV>'] = set()

        self.collect_test_statistics()
        # moved to the MStep


    def doMStep(self, new_transition_counts=None, new_emission_counts=None,
            new_state_counts=None, new_observations=None, new_tag_dict=None):

        # Estimate model parameters
        # 1) Reset all counts to original from training
        # 2) Add counts of known train and estimated new counts.
        # 3) Recalculate smoothed probabilities.

        self.transition_counts = self.org_transition_counts
        self.emission_counts = self.org_emission_counts
        self.observations = self.org_observations
        self.state_counts = self.org_state_counts
        self.tag_dict = self.org_tag_dict

        self.emission_probs = {}
        self.transition_probs = {}

        # Update transition counts
        if not new_transition_counts is None:
            
            for prev in new_transition_counts.keys():
                for curr in new_transition_counts[prev].keys():
                    count = new_transition_counts[prev][curr]
                    # add count function handles new items
                    self.transition_counts = add_count(count, prev, curr,
                            self.transition_counts)
                    
        # Update emission counts
        if not new_emission_counts is None:

            for tag in new_emission_counts.keys():
                for word in new_emission_counts[tag].keys():
                    count = new_emission_counts[tag][word]
                    # add count function handles new items
                    self.emission_counts = add_count(count, tag, word, self.emission_counts)

        # Update observations
        if not new_observations is None:
            self.observations = self.observations.union(new_observations)


        self.calc_smooth_probs(self.transition_counts, self.transition_probs, self.states)
        self.calc_smooth_probs(self.emission_counts, self.emission_probs, self.observations)
        #self.calc_smooth_probs(self.org_sfx_emission_counts, self.sfx_emission_probs, self.sfx_observations)

        
        # Update tag_dict
        # This step is abit wasteful because we keep seeing the same words anyway
        if not new_tag_dict is None:
            for word in new_tag_dict.keys():
                if word in self.tag_dict:
                    self.tag_dict[word] = self.tag_dict[word].union(new_tag_dict[word])
                else:
                    self.tag_dict[word] = new_tag_dict[word]



        # Update state counts
        if not new_state_counts is None:
            for tag in new_state_counts.keys():
                self.state_counts[tag] += new_state_counts[tag]


        # Finally Handle OOV
        for tag in self.state_counts.keys():
            self.tag_dict['<OOV>'].add(tag)
            tag_total = sum(self.state_counts.values())

            self.emission_probs[tag]['<OOV>'] = \
                    float(self.state_counts[tag]) / tag_total

#    def doEStep():
        # EStep consist of forward backward
        # but we need to know whether we are doing it for raw or test
        # if EStep, we are doing forward-backward on raw

#        raw_contents 
#        calc_state_probs


    def calc_state_probs(self, i, w_i, prev_word, direction=None):
        # if forward, state_probs = self.alpha_t
        # if backward, state_probs = self.beta_t

        if direction =="forward":
            state_p = self.alpha_t
        
        elif direction == "backward":
            state_p = self.beta_t

        tag_d = self.tag_dict
        emission_p = self.emission_probs

        # TODO: handle EOS
        if w_i == "###":
            # SOS
            if prev_word == '':
                if direction == "forward":
                    self.mu_t[w_i] = [math.log(1)]
                prev_word = "###"
                return prev_word
            
            # EOS
            else:
                tags = {'###'}
                prev_tags = tag_d[prev_word]
           
        # First word
        elif prev_word == '###':
            tags = tag_d[w_i]
            prev_tags = {'###'} # ### tags ###

        else:
            tags = tag_d[w_i]
            prev_tags = tag_d[prev_word]
        
        for t_i in tags:
            
            # VITERBI
            if direction == "forward":
                if t_i not in self.mu_t.keys():
                    self.mu_t[t_i] = [math.log(1)]
                while len(self.mu_t[t_i]) <= i:
                    self.mu_t[t_i].append(float('-inf'))
                if t_i not in self.back_t.keys():
                    self.back_t[t_i] = ['###']
                while len(self.back_t[t_i]) <= i:
                    self.back_t[t_i].append('')


            for t_im1 in prev_tags:
                p = math.log(self.transition_probs[t_im1][t_i]) + \
                    math.log(emission_p[t_i][w_i])

                mu = state_p[t_im1][i-1] + p

                if direction == "forward" and mu > self.mu_t[t_i][i]:
                    self.mu_t[t_i][i] = mu
                    self.back_t[t_i][i] = t_im1

                # Cannot just add log probs here. 
                # we need to do log sum exponentials as explained on R-10
                state_p[t_i][i] = np.logaddexp(state_p[t_i][i], mu)


    def get_max_tag(self, i, w_i):

        if w_i=="###":
            return "###"

        tag_d = self.tag_dict

        max_tag_score = float('-inf')
        max_tag = None

        #for tag in self.org_tag_dict[w_i]:
        for tag in tag_d[w_i]:
            predicted_tag_score = self.alpha_t[tag][-i-1] + self.beta_t[tag][i]
            
            # Line 13 of pseudocode normalises but there is no need to normalise since we are
            # taking the max.

            if predicted_tag_score > max_tag_score:
                max_tag = tag
                max_tag_score = predicted_tag_score

        if max_tag is None:
            max_tag = 'N'
        return max_tag


    def evaluate(self, pred_tags, true_tags, known):
        correct = 0
        total = 0
        known_total = 0
        known_correct = 0
        unknown_correct = 0

        wrong_tags = []

        for i in range(0, len(true_tags)):
            if true_tags[i] == pred_tags[i] and true_tags[i] != "###":
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

                wrong_tags.append(true_tags[i])

        unknown_total = total - known_total
        accuracy = float(correct) / total
        known_accuracy = float(known_correct) / known_total
        if unknown_total > 0:
            unknown_accuracy = float(unknown_correct) / unknown_total
        else:
            unknown_accuracy = 0

        print("Wrongly tagged:", Counter(wrong_tags))
        return (accuracy, known_accuracy, unknown_accuracy)


#    def doEStep(self, mode="test"):

    def collect_test_statistics(self):

        self.true_tags = []
        self.test_words = []

        self.known = [True] * len(self.testlines)

        for i in range(len(self.testlines)):
            line = self.testlines[i]
            w_i, tag = line.strip().split('/')

            if w_i not in self.org_observations:
            
                self.known[i] = False
            #    w_i = "<OOV>"

            self.true_tags.append(tag)
            self.test_words.append(w_i)


    def test_and_eval(self):
            
        # Forward backward
        predicted_tags, crossentropy = self.forward_backward(mode="test")

        # Viterbi evaluation
        n = len(self.testlines)-1
        viterbi_tags = [''] * (n+1)
        viterbi_tags[n] = '###'

        for i in range(n, -1, -1):
            viterbi_tags[i-1] = self.back_t[viterbi_tags[i]][i]
        
        viterbi_eval = self.evaluate(viterbi_tags, self.true_tags, self.known)
 
        # Write test output
        #
        with open('test-output', 'w') as out_f:
            for i, word in enumerate(self.test_words):
                out_f.write(word+'/'+predicted_tags[i]+'\n')

        # evaluate posterior decoding
        posterior_eval = self.evaluate(predicted_tags, self.true_tags, self.known)
        
        # Calculate perplexity at the end of forwardpass
        # length -1 because of ###/### at SOS
        # 
        # self.alpha_t['###'] should contain the sum of alpha_t
        perplexity = math.exp( - crossentropy / (len(self.testlines)-1))
        print("Model perplexity per tagged test word:", perplexity)
        print("Tagging accuracy (Viterbi decoding): {0:.2f}%\t(known: {1:.2f}% novel: {2:.2f}%)".format(\
                viterbi_eval[0]*100, viterbi_eval[1]*100, viterbi_eval[2]*100))
        print("Tagging accuracy (posterior decoding): {0:.2f}%\t(known: {1:.2f}% novel: {2:.2f}%)".format(\
                posterior_eval[0]*100, posterior_eval[1]*100, posterior_eval[2]*100))



    def forward_backward(self, mode="test"):
        ### VITERBI
        # Based on the algorithm on page R-4 in handout
        self.mu_t = {}
        prev_word = ''
        self.back_t = {}

        self.alpha_t = {}
        self.beta_t = {}
        
        # Similar to back_t in viterbi
        predicted_tags = []

        crossentropy = 0

        if mode=="test":
            words = self.test_words
        elif mode=="raw":
            words = self.raw_words


        # initialize alpha_t and beta_t
        for state in self.states:
            self.alpha_t[state] = [math.log(1)]
            self.beta_t[state] = [math.log(1)]

            for i in range(1, len(self.testlines)):
                self.alpha_t[state].append(float('-inf'))
                self.beta_t[state].append(float('-inf'))

        
        # forward pass
        for i, w_i in enumerate(words):
            if w_i not in self.observations:
            #    known[i] = False
                w_i = "<OOV>"

            if i > 0:
                crossentropy += math.log(self.transition_probs[self.true_tags[i-1]][self.true_tags[i]]) \
                    + math.log(self.emission_probs[self.true_tags[i]][w_i])

            self.calc_state_probs(i, w_i, prev_word, direction="forward")
            prev_word = w_i

        # backward pass
        #   note beta_t is reversed indexed in order to reuse the way initialisation works 
        #   i.e., beta_t[0] is the state of the last observed word.
        #   hence predicted_tags is reversed indexed as well.

        prev_word = ''
        
        for i in range(len(words)):

            w_i = words[len(words)-1-i]
            if w_i not in self.observations:
                w_i = "<OOV>"

            self.calc_state_probs(i, w_i, prev_word, direction="backward")
            prev_word = w_i

            predicted_tags.append(self.get_max_tag(i, w_i))

            # append counts here.
            # to be used in the M step
        
        tags = predicted_tags[::-1]
        return tags, crossentropy





# run Hidden Markov Model on specified training and testing files
#hmm = HMM()
#hmm.train(sys.argv[1])
#hmm.runEM()
#hmm.test_and_eval(sys.argv[2])
