"""Deep Pyramid Convolutional Neural Network"""

from keras import Model, Input
from keras.layers import MaxPooling1D, Conv1D, BatchNormalization, PReLU, \
    add, GlobalMaxPool1D, Dense, Dropout, Embedding
from keras.regularizers import l2

MANDATORY_ARGS = ['words_in_corpus', 'output_size', 'embedding_matrix']

DEFAULT_PARAMS = {
    # Mandatory parameters
    'words_in_corpus': None,
    'output_size': None,
    'embedding_matrix': None,

    # Hyperparameter defaults
    'trainable_embeddings': False,
    'embedding_size': 300,
    'max_sentence_length': 140,  # words
    'num_of_filters': 256,
    'dense_hidden_size': 256,
    'embedding_dropout_rate': 0.4,
    'num_of_dpcnn_blocks': 3,
    'dense_dropout_rate': 0.3,
    'max_pool_size': 3,
    'max_pool_strides': 2,
    'conv_kernel_size': 3,
    'kernel_regularizer_strength': 0.01
}


def _dpcnn_block(params):

    def build_dpcnn_block(x):
        x = MaxPooling1D(
            pool_size=params.get('max_pool_size'),
            strides=params.get('max_pool_strides'))(x)

        block = Conv1D(filters=params.get('num_of_filters'),
                       kernel_size=params.get('conv_kernel_size'),
                       padding='same',
                       activation='linear',
                       kernel_regularizer=l2(
                           params.get('kernel_regularizer_strength'))
                       )(x)
        block = BatchNormalization()(block)
        block = PReLU()(block)

        block = Conv1D(filters=params.get('num_of_filters'),
                       kernel_size=params.get('conv_kernel_size'),
                       padding='same',
                       activation='linear',
                       kernel_regularizer=l2(
                           params.get('kernel_regularizer_strength'))
                       )(block)
        block = BatchNormalization()(block)
        block = PReLU()(block)

        x = add([block, x])  # skip connection

        return x

    return build_dpcnn_block


def _classification_block(outputs_num, dense_hidden_size, dense_dropout_rate):
    def build_classification_block(x):
        output = GlobalMaxPool1D()(x)

        # two dense blocks w/ PReLU in between
        output = Dense(dense_hidden_size, activation='linear')(output)
        output = BatchNormalization()(output)
        output = PReLU()(output)

        output = Dropout(dense_dropout_rate)(output)
        output = Dense(outputs_num, activation='sigmoid')(output)

        return output

    return build_classification_block


def _check_mandatory_params(kwargs):
    not_present = [arg not in kwargs for arg in MANDATORY_ARGS]

    if any(not_present):
        missing_args = filter(lambda item: item[0], zip(not_present, MANDATORY_ARGS))
        missing_arg_names = [missing[1] for missing in missing_args]

        raise ValueError(f'Missing required args: {missing_arg_names}')


def build_dpcnn_model(**kwargs) -> Model:
    """Builds and returns the DPCNN model"""

    _check_mandatory_params(kwargs)
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    input_layer = Input(shape=(params.get('max_sentence_length'),), dtype='int32')

    embedding_layer = Embedding(
        input_dim=params.get('words_in_corpus') + 1,
        output_dim=params.get('embedding_size'),
        weights=[params.get('embedding_matrix')],
        trainable=params.get('trainable_embeddings')
    )(input_layer)

    dropout = Dropout(params.get('embedding_dropout_rate'))(embedding_layer)

    x = Conv1D(filters=params.get('num_of_filters'),
               kernel_size=params.get('conv_kernel_size'),
               padding='same',
               activation='linear',
               kernel_regularizer=l2(
                   params.get('kernel_regularizer_strength'))
               )(dropout)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv1D(filters=params.get('num_of_filters'),
               kernel_size=params.get('conv_kernel_size'),
               padding='same',
               activation='linear',
               kernel_regularizer=l2(
                   params.get('kernel_regularizer_strength'))
               )(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    if params.get('embedding_size') == params.get('num_of_filters'):
        # no shape matching required
        x = add([dropout, x])
    else:
        # TODO perform shape matching for the first skip connection
        # Just skip for now :)
        pass

    for _ in range(params.get('num_of_dpcnn_blocks')):
        x = _dpcnn_block(params)(x)

    predictions = _classification_block(
        params.get('output_size'),
        params.get('dense_hidden_size'),
        params.get('dense_dropout_rate'))(x)

    model = Model(inputs=input_layer, outputs=predictions)
    return model
