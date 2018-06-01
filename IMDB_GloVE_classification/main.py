#!/usr/bin/python
# encoding: utf-8
"""
main
Purpose: Binary Classification with IMDB dataset using GloVe Embedding
Data: IMDB dataset (imported with Keras)
Input: 50,000 reviews of movies, labeled with 0 [negativ rating] or 1 [positiv rating]
Model: Sequential (Embedding + Dense), developed with Keras
Output: Probability of positiv rating -> Classification 1 [positiv rating] if proba>0.5 else 0 [negativ rating]

Author: datadonk23 (datadonk23@gmail.com)
Date: 2018-06-01
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend


def main():
    """
    Main Loop

    :return: -
    """
    #Setup
    backend.clear_session()




if __name__ == "__main__":
    main()