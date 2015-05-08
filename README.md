# Synonym_Extraction

Use Three models for synonym extraction. 
Implement the algorithm described in http://arxiv.org/pdf/1412.2197.pdf

#Pipeline:
1) Use Wordnet to generate: 22k synonyms, 2k antonyms, 80k irrelevant words.

2) Train word vector, each word is mapped to a 50d vector

3) Generate Training data: 
Synonym with label 1, Antonym/Irrelevant with label -1. For each pair of words, feature is of 
250d, a concatenation of x1, x2, x1*x2, |x1-x2|, x1+x2
4) Split data to Train/Test , 33% training data and 67% test data
5) Run three classifiers on the data: svm, neural network, deep neural network.


#Model
1) SVM: with C=1.0

2) Multilayer Perceptron: 
  Input Layer: 250
  Hidden Layer 100, with tanh activation
  Ouput Layer 1

3) Deep MLP:
  Input Layer: Tensor of 50*3
  Layer without bias term: size 3*10
  Flatten Layer: Transform the 50*10 2d tensor to 500d vector
  Hidden Layer: 500*100, tanh activation
  Output Layer: 100*1, None activation
