'''
Name: Aaron Mueller & Suzanna Sia
Course: EN.601.665 -- Natural Language Processing
Instructor: Jason Eisner
Date: 15 Nov. 2018
Assignment: HW 6 -- HMMs
'''

import sys
import pdb
from collections import defaultdict

class HMM:
    def __init__(self):
        self.states = set()
        self.observations = set()
        self.start_probs = {}
        self.transition_probs = {}
        self.emission_probs = {}
        self.tag_dict = {}

    # fill variables declared in __init__
    def train(self, train_file):
        with open(train_file, 'r') as f:
            start_counts = defaultdict(int)
            num_sentences = 0
            num_words = 0
            transition_counts = {}
            emission_counts = {}
            
            SOS = False

            for line_no, line in enumerate(f):
                word, tag = line.strip().split('/')

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
                # Hardcoded for now
                if word not in self.tag_dict.keys():
                    self.tag_dict[word] = ['H', 'C']
                    #self.tag_dict[word] = []
                
                #if tag not in self.tag_dict[word]:
                #    self.tag_dict[word].append(tag)
                

                #if word not in self.observations:
                #    self.observations.add(word)
                #if tag not in self.states:
                #    self.states.add(tag)
                
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

                for tag in transition_counts[prev_tag]:

                    self.transition_probs[prev_tag][tag] = \
                            float(transition_counts[prev_tag][tag]) / transition_sum
            
            
            #for word in emission_counts:
            #    self.emission_probs[word] = {}
            #    emission_sum = sum(emission_counts[word].values())
            #    for tag in emission_counts[word]:
            #        self.emission_probs[word][tag] = \
            #                float(emission_counts[word][tag]) / emission_sum


            for tag in emission_counts:
                self.emission_probs[tag] = {}
                emission_sum = sum(emission_counts[tag].values())
                for word in emission_counts[tag]:
                    self.emission_probs[tag][word] = \
                            float(emission_counts[tag][word]) / emission_sum

        print("\ntransition probabilities:", self.transition_probs)

        print("\nemission probabilities:", self.emission_probs)
        #print("\nstart probabilities:", self.start_probs)
        
        print()


    def test_and_eval(self, test_file):
        ### VITERBI
        # Based on the algorithm on page R-4 in handout
        #mu_t = [1]
        mu_t = []
        prev_word = ''
        back_t = []
        #back_t = ['']

        true_tags = []

        with open(test_file, 'r') as f:

            for i, line in enumerate(f):
                #i = line_no + 1
                w_i, tag = line.strip().split('/')
                true_tags.append(tag)
    
                # TODO: handle EOS
                if w_i == "###":
                    
                    # SOS
                    if prev_word == '':
                        back_t.append('###')
                        mu_t.append(1) # the probability of starting is 1
                        prev_word = "###"
                        continue
                    
                    # EOS
                    else:
                        tags = {'###'}
                        prev_tags = self.tag_dict[prev_word]
                
                # First word
                elif prev_word == '###':
                    tags = self.start_probs.keys()
                    prev_tags = {'###'} # ### tags ###

                else:
                    tags = self.tag_dict[w_i]
                    prev_tags = self.tag_dict[prev_word]
                
                # initialize mu_t[i] and back_t[i]
                mu_t.append(0)
                back_t.append('')


                for t_i in tags:
                    for t_im1 in prev_tags:
                        
                        p = hmm.transition_probs[t_im1][t_i] * hmm.emission_probs[t_i][w_i]
                        mu = mu_t[i-1] * p

                        if mu > mu_t[i]:
                            mu_t[i] = mu
                            back_t[i] = t_i

                # Sanity check against Viterbi excel sheet
                # running into underflow issues?
                if (i<15):
                    print(i, back_t[i], mu_t[i])


                prev_word = w_i  
            
            '''
            # go through backpointers, recover most likely tags
            pred_tags = []
            for i in range(len(back_t)-1, -1, -1):
                pred_tags.append(back_t[i])
            
            # reverse array to get sequence of tags in sentence order
            pred_tags = pred_tags[::-1]
            '''

            # compare and evaluate
            correct = 0
            total = 0
            for i in range(0, len(true_tags)):
                if true_tags[i] == back_t[i] and true_tags != "###":
                    correct += 1
                    total += 1
                elif true_tags != "###":
                    total += 1
            
            print("accuracy: {}".format(float(correct) / total))
            print("predicted tags: {}\nlength: {}".format(back_t, len(back_t)))
            print("true tags: {}\nlength: {}".format(true_tags, len(true_tags)))     
    

# run Hidden Markov Model on specified training and testing files
hmm = HMM()
hmm.train(sys.argv[1])
hmm.test_and_eval(sys.argv[2])
