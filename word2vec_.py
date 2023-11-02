import argparse
import numpy as np
import random

from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    s = 1/ (1+ np.exp(-x))

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """
    
    y = 1
    forwardPath = np.dot(outsideVectors, centerWordVec)
    outputProb = softmax(forwardPath)
    loss = - np.log(outputProb[outsideWordIdx])

    backwardPath = outputProb
    backwardPath[outsideWordIdx] -= y
    backwardPath = backwardPath[:,np.newaxis]

    gradCenterVec = np.dot(outsideVectors.T, backwardPath)

    gradOutsideVecs = np.dot(backwardPath, centerWordVec[:, np.newaxis].T)

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    outsideVectors_pos = outsideVectors[outsideWordIdx]
    outsideVectors_neg = outsideVectors[negSampleWordIndices]

    sigmoid_pos = sigmoid(np.dot(outsideVectors_pos, centerWordVec.T))
    # print(sigmoid_pos.shape)
    sigmoid_neg = sigmoid(-np.dot(outsideVectors_neg, centerWordVec))
    # print(sigmoid_neg.shape)
    log_pos = np.log(sigmoid_pos)
    log_neg = np.log(sigmoid_neg)

    loss = -log_pos - np.sum(log_neg)
    # print(np.sum((1-sigmoid_neg[:, np.newaxis])*  outsideVectors_neg, axis = 0)[:, np.newaxis].shape)
    gradCenterVec = -(np.dot((1-sigmoid_pos),outsideVectors_pos))[:, np.newaxis] + np.sum((1-sigmoid_neg[:, np.newaxis])*  outsideVectors_neg, axis = 0)[:, np.newaxis]
    # print(gradCenterVec.shape)
    gradOutsideVecs = np.zeros_like(outsideVectors)

    gradOutsideVecs[outsideWordIdx] = - (1- sigmoid_pos)* centerWordVec
    # print(negSampleWordIndices)
    # [3, 3, 0, 4, 0, 2, 0, 2, 4, 4]
    i = 0
    for idx in negSampleWordIndices:
        gradOutsideVecs[idx] += (1-sigmoid_neg[i]) * centerWordVec
        i += 1

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)
    index_center_word = word2Ind[currentCenterWord]
    index_ouside_word = [word2Ind[i] for i in outsideWords]
    
    center_word_vector = centerWordVectors[index_center_word]
    # outside_word_vector = outsideVectors[index_ouside_word]
    for index in index_ouside_word:
        loss_temp, gradCenterVecs_temp, gradOutsideVectors_temp = word2vecLossAndGradient(center_word_vector, index, outsideVectors, dataset)

        loss += loss_temp
        gradCenterVecs[index_center_word][:, np.newaxis] += gradCenterVecs_temp
        gradOutsideVectors += gradOutsideVectors_temp
    
    return loss, gradCenterVecs, gradOutsideVectors


def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


