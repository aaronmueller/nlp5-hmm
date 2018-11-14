# Aaron Mueller, Suzanna Sia
## Tagging with Hidden Markov Models

TODO:
* write `vtag` for ice cream data (2)
	* bigram Viterbi tagger
	* unsmoothed estimates
* improve `vtag` for use on real data (3)
	* smoothed probabilities
	* tag dictionary
* extend `vtag` to use posterior decoding
	* create a `test-output` file which contains tagging of test data
* improve HMM tagger to do well on leaderboard
	* explain how much tagger improved accuracy and perplexity of baseline tagger with observations and results
	* save output of running `vtag entrain entest`
* write `vtagem`
	* copy `vtag` to `vtagem`, modify to reestimate parameters of HMM on untagged data
	* forward-backward algorithm
	* 10 iterations of EM
	* submit source code for `vtagem` and output of `vtagem entrain25k entest enraw`
	* answer questions from HW
* speed up `vtagem` for czech
	* answer questions from HW
