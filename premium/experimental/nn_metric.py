#!/usr/bin/env python
import os
import random
import re
import sys

import codefast as cf
import joblib
import numpy as np 
import pandas as pd


class MatMul(object):
    def __init__(self, W) -> None:
        self.params=[W]
        self.grads=[np.zeros_like(W)]
        self.x=None
    
    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x=x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
class SimpleCBOW(object):
    def __init__(self, vocab_size:int, hidden_size:int):
        V,H=vocab_size,hidden_size
        W_in=np.random.randn(V,H).astype(np.float32) * 0.01
        W_out=np.random.randn(H,V).astype(np.float32) * 0.01
        self.in_layer0=MatMul(W_in)
        self.in_layer1=MatMul(W_in)
        self.out_layer0=MatMul(W_out)
        self.loss_layer=SoftmaxWithLoss()

        layers=[self.in_layer0,self.in_layer1,self.out_layer0]
        self.params,self.grads=[],[]
        for l in layers:
            self.params.extend(l.params)
            self.grads.extend(l.grads)
        self.word_vectors=W_in

def softmax(x):
    if x.ndim == 1:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]

class SoftmaxWithLoss(object):
    def __init__(self):
        self.params,self.grads=[],[]
        self.y=None
        self.t=None
    def forward(self, x, t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size=self.t.shape[0]
        if self.y.ndim==1:
            dx=-(self.t-self.y)
        else:
            dx=self.y.copy()
            dx[np.arange(batch_size),self.t]=-1
        dx=dx/batch_size
        self.grads[0][...]=dx
        return dx
