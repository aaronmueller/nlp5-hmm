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

NUM_SET = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

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
<<<<<<< Updated upstream
=======

    def get_suffix(self, word):
        #
        # uncomment "return word" below to remove the effect of suffix
        #
        
        # return word

        # -4 is the best length so far
        # from 92.65% to 93.59%

        #if len(word)>3:
        #    sfx = word[-4:]
        if len(word)>3:
            sfx = word[-4:]
        else:
            sfx = word
        return sfx
    
    def add_count(self, ind1, ind2, count_dict):
        if ind1 not in count_dict:
            count_dict[ind1] = defaultdict(int)
        count_dict[ind1][ind2] +=1
        return count_dict

    def calc_smooth_probs(self, count_dict, probs, vals, lamda):

        #count_dict = transition_counts
        #probs = transition_probs
        #target = self.states

        for index1 in count_dict:
            probs[index1] = {}
            normalising_sum = sum(count_dict[index1].values())

            for index2 in vals:
                if index2 in count_dict[index1]:
                    probs[index1][index2] = \
                        float(count_dict[index1][index2] + lamda)/ \
                        (normalising_sum + (lamda * len(vals)))
                else:
                    probs[index1][index2] = \
                        lamda/ (normalising_sum + lamda * len(vals))

>>>>>>> Stashed changes
    def train(self, train_file):
        with open(train_file, 'r') as f:
            start_counts = defaultdict(int)
            num_sentences = 0
            num_words = 0
            transition_counts = {}
            emission_counts = {}
            state_counts = defaultdict(int)
            
            SOS = False

            lamda = 1
            self.observations.add('<OOV>')
            self.observations.add('<NUM>')
            self.tag_dict['<OOV>'] = set()

            for line_no, line in enumerate(f):
                word, tag = line.strip().split('/')
<<<<<<< Updated upstream
=======

                if word[0] in NUM_SET and word[-1] in NUM_SET:
                    word = '<NUM>'
                
                sfx = self.get_suffix(word)
>>>>>>> Stashed changes

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
               
                state_counts[tag] += 1

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

<<<<<<< Updated upstream
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

=======

                               
            self.calc_smooth_probs(transition_counts, self.transition_probs, self.states, lamda)
            self.calc_smooth_probs(emission_counts, self.emission_probs, self.observations, lamda)
            self.calc_smooth_probs(sfx_emission_counts, self.sfx_emission_probs,
                    self.sfx_observations, lamda)

            
>>>>>>> Stashed changes
            for tag in state_counts.keys():
                self.tag_dict['<OOV>'].add(tag)
                tag_total = sum(state_counts.values())

                self.emission_probs[tag]['<OOV>'] = \
                        float(state_counts[tag]) / tag_total


        # print("transition probabilities:", self.transition_probs)
        # print("emission probabilities:", self.emission_probs)
        

    def calc_state_probs(self, i, w_i, prev_word, direction=None):
        # if forward, state_probs = self.alpha_t
        # if backward, state_probs = self.beta_t

        if direction =="forward":
            state_p = self.alpha_t
        
        elif direction == "backward":
            state_p = self.beta_t

        # TODO: handle EOS
        if w_i == "###":
            # SOS
            if prev_word == '':
                state_p[w_i] = [math.log(1)] # the probability of starting is 1
                if direction == "forward":
                    self.mu_t[w_i] = [math.log(1)]
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
            if t_i not in state_p.keys():
                state_p[t_i] = [math.log(1)]
            
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

            while len(state_p[t_i]) <= i:
                state_p[t_i].append(float('-inf'))

            for t_im1 in prev_tags:
                p = math.log(self.transition_probs[t_im1][t_i]) + \
                    math.log(self.emission_probs[t_i][w_i])
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

        max_tag_score = float('-inf')
        max_tag = None

        for tag in self.tag_dict[w_i]:
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

        unknown_total = total - known_total
        accuracy = float(correct) / total
        known_accuracy = float(known_correct) / known_total
        if unknown_total > 0:
            unknown_accuracy = float(unknown_correct) / unknown_total
        else:
            unknown_accuracy = 0


        return (accuracy, known_accuracy, unknown_accuracy)


    def test_and_eval(self, test_file):
        ### VITERBI
        # Based on the algorithm on page R-4 in handout
        #self.mu_t = [1]
        self.mu_t = {}
        prev_word = ''
        self.back_t = {}

        self.alpha_t = {}
        self.beta_t = {}
        #self.back_t = ['']
        
        # Similar to back_t in viterbi
        predicted_tags = []
        true_tags = []

        crossentropy = 0

        with open(test_file, 'r') as f:
            testlines = f.readlines()
        
        known = [True] * len(testlines)

        # initialize alpha_t and beta_t
        for state in self.states:
            self.alpha_t[state] = [math.log(1)]
            self.beta_t[state] = [math.log(1)]

            for i in range(1, len(testlines)):
                self.alpha_t[state].append(float('-inf'))
                self.beta_t[state].append(float('-inf'))

        # forward pass
        for i in range(len(testlines)):
            line = testlines[i]

            w_i, tag = line.strip().split('/')
            true_tags.append(tag)

            if w_i[0] in NUM_SET and w_i[-1] in NUM_SET:
                w_i = '<NUM>'

            if w_i not in self.observations:
                w_i = '<OOV>'
                known[i] = False

            if i > 0:
                crossentropy += math.log(self.transition_probs[true_tags[i-1]][true_tags[i]]) \
                        + math.log(self.emission_probs[tag][w_i])

            self.calc_state_probs(i, w_i, prev_word, direction="forward")
            prev_word = w_i

        # compare and evaluate
        # viterbi
        n = i
        tags = [''] * (n+1)
        tags[n] = '###'
        for i in range(n, -1, -1):
            tags[i-1] = self.back_t[tags[i]][i]
        
        viterbi_eval = self.evaluate(tags, true_tags, known)

        # backward pass
        #   note beta_t is reversed indexed in order to reuse the way initialisation works 
        #   i.e., beta_t[0] is the state of the last observed word.
        #   hence predicted_tags is reversed indexed as well.

        prev_word = ''
        self.mu_t = {}
        self.back_t = {}
        
        original_words = []
        for i in range(len(testlines)):
            line = testlines[len(testlines)-1-i]
            w_i, tag = line.strip().split('/')
            original_words.append(w_i)

<<<<<<< Updated upstream
=======
            if w_i[0] in NUM_SET and w_i[-1] in NUM_SET:
                w_i = '<NUM>'
            
>>>>>>> Stashed changes
            if w_i not in self.observations:
                w_i = '<OOV>'

            self.calc_state_probs(i, w_i, prev_word, direction="backward")
            prev_word = w_i

            predicted_tags.append(self.get_max_tag(i, w_i))

        # Sanity check
        #for i in range(len(self.beta_t['H'])):
        #    print(i, math.exp(self.beta_t['H'][i]))

        
        tags = predicted_tags[::-1]
        original_words = original_words[::-1]

        with open('test-output', 'w') as out_f:
            for i, word in enumerate(original_words):
                out_f.write(word+'/'+tags[i]+'\n')

        # evaluate posterior decoding
        posterior_eval = self.evaluate(tags, true_tags, known)
        
        # Calculate perplexity at the end of forwardpass
        # length -1 because of ###/### at SOS
        # 
        # self.alpha_t['###'] should contain the sum of alpha_t
        perplexity = math.exp( - crossentropy / (len(testlines)-1))
        print("Model perplexity per tagged test word:", perplexity)
        print("Tagging accuracy (Viterbi decoding): {0:.2f}%\t(known: {1:.2f}% novel: {2:.2f}%)".format(\
                viterbi_eval[0]*100, viterbi_eval[1]*100, viterbi_eval[2]*100))
        print("Tagging accuracy (posterior decoding): {0:.2f}%\t(known: {1:.2f}% novel: {2:.2f}%)".format(\
                posterior_eval[0]*100, posterior_eval[1]*100, posterior_eval[2]*100))





# run Hidden Markov Model on specified training and testing files
hmm = HMM()
hmm.train(sys.argv[1])
hmm.test_and_eval(sys.argv[2])
