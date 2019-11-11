POS Tagging using 2 approaches.

1. Viterbi algorithm with a hidden markov model

2. A Convolutional Neural Network over character embeddings, combined with an LSTM for tag prediction.

Neither approach is optimal, e.g. Consider dynamic suffix lengths for the viterbi algorithm, and batch LSTM training for the neural network approach.