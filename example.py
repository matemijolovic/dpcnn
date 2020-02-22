"""Example usage of the dpcnn model"""

import numpy as np

from model.dpcnn import build_dpcnn_model

if __name__ == '__main__':
    model = build_dpcnn_model(
        words_in_corpus=10,
        output_size=10,
        embedding_matrix=np.zeros(shape=(11, 300)))  # Index 0 reserved for OOV
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy')
    model.summary(line_length=150)
    # use model as a regular Keras model
