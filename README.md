# MEMM-Viterbi

My implementation of the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) using the [MEMM](https://en.wikipedia.org/wiki/Maximum-entropy_Markov_model) model for the NLP sequence tagging problem.

Implemented in Python3 and uses `sklearn` implementation of `LogisticRegression` for training and predictions. Exposes one class: `MemmViterbi` that has two public APIs: `train` for training the model and `predict` for predicting the sequence tag of the given sentence.
