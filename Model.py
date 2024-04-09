from keras.layers import *
from keras.models import *
from keras.regularizers import l1, l2
import numpy as np

from multi_head_att import MultiHeadAttention

from keras import backend as K
from keras.layers import Layer

       
class InS_mechanism(Layer):
    """
       Attentive Matching strategy, each contextual embedding is compared with its attentive
       weighted representation of the other sentence.
       From Bilateral Multi-Perspective Matching for Natural Language Sentences(https://arxiv.org/abs/1702.03814)
    """
    def __init__(self, perspective_num=10, **kwargs):
        self.perspective_num = perspective_num
        self.kernel = None
        super(InS_mechanism, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.max_len = input_shape[0][1]
        self.kernel = self.add_weight(name='kernel', shape=(self.perspective_num, self.dim),
                                      initializer='glorot_uniform')
        super(InS_mechanism, self).build(input_shape)

    def _cosine_similarity(self, x1, x2):
        cos = K.sum(x1 * x2, axis=-1)
        x1_norm = K.sqrt(K.maximum(K.sum(K.square(x1), axis=-1), K.epsilon()))
        x2_norm = K.sqrt(K.maximum(K.sum(K.square(x2), axis=-1), K.epsilon()))
        cos = cos / x1_norm / x2_norm
        return cos
        
    def _mean_attentive_vectors(self, x2, cosine_matrix):
        """Mean attentive vectors.
        Calculate mean attentive vector for the entire sentence by weighted
        summing all the contextual embeddings of the entire sentence
        # Arguments
            x2: sequence vectors, (batch_size, x2_timesteps, embedding_size)
            cosine_matrix: cosine similarities matrix of x1 and x2,
                           (batch_size, x1_timesteps, x2_timesteps)
        # Output shape
            (batch_size, x1_timesteps, embedding_size)
        """
        # (batch_size, x1_timesteps, x2_timesteps, 1)
        expanded_cosine_matrix = K.expand_dims(cosine_matrix, axis=-1)
        # (batch_size, 1, x2_timesteps, embedding_size)
        x2 = K.expand_dims(x2, axis=1)
        # (batch_size, x1_timesteps, embedding_size)
        weighted_sum = K.sum(expanded_cosine_matrix * x2, axis=2)
        # (batch_size, x1_timesteps, 1)
        sum_cosine = K.expand_dims(K.sum(cosine_matrix, axis=-1) + K.epsilon(), axis=-1)
        # (batch_size, x1_timesteps, embedding_size)
        attentive_vector = weighted_sum / sum_cosine
        return attentive_vector

    def call(self, inputs, **kwargs):
        
        sent1 = inputs[0]
        sent2 = inputs[1]

        x1 = K.expand_dims(sent1, axis=2)
        x2 = K.expand_dims(sent2, axis=1)
        cos_matrix  = self._cosine_similarity(x1, x2)

        v1 = K.expand_dims(sent1, -2) * self.kernel
        
        mean_attentive_vec = self._mean_attentive_vectors(sent2, cos_matrix)
        mean_attentive_vec = K.expand_dims(mean_attentive_vec, -2) * self.kernel
        matching = self._cosine_similarity(v1, mean_attentive_vec)
        return matching

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.perspective_num

    def get_config(self):
        config = {'perspective_num': self.perspective_num}
        base_config = super(InS_mechanism, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
                                                                                                                                                                                                                                                                                                                       

def DeepPepPI_model(pept_emb_shape, prot_mat_shape):
    # DCR module
    pept_emb_input =  Input(shape = (pept_emb_shape[1], pept_emb_shape[2]))
    pept_cnn = Conv1D(filters=128,kernel_size=4)(pept_emb_input)
    pept_cnn_act = LeakyReLU(alpha = 0.02)(pept_cnn)
    pept_maxpool = MaxPooling1D(pool_size=3)(pept_cnn_act)
    pept_maxpool_D = Dropout(0.25)(pept_maxpool)
    pept_LSTM = Bidirectional(LSTM(128, return_sequences=True))(pept_maxpool_D)
    pept_LSTM_D = Dropout(0.25)(pept_LSTM)

    # BSS module
    prot_mat_input = Input(shape = (prot_mat_shape[1], prot_mat_shape[2]))
    prot_cnn = Conv1D(filters=64,kernel_size=3)(prot_mat_input)
    prot_cnn_act = LeakyReLU(alpha = 0.02)(prot_cnn)
    prot_max_pool = MaxPooling1D(pool_size=2)(prot_cnn_act)
    prot_maxpool_D = Dropout(0.25)(prot_max_pool)
    prot_LSTM = Bidirectional(LSTM(128, return_sequences=True))(prot_maxpool_D)
    prot_LSTM_D = Dropout(0.25)(prot_LSTM)

    # MSA Mechanism
    pept_MSA = MultiHeadAttention(8)(pept_LSTM)
    pept_MSA_global_pool = GlobalAveragePooling1D()(pept_MSA)
    prot_MSA = MultiHeadAttention(8)(prot_LSTM)
    prot_MSA_global_pool = GlobalAveragePooling1D()(prot_MSA)

    # InS Mechanism
    InS1 = InS_mechanism(64)([pept_LSTM, prot_LSTM])
    InS1_global_pool = GlobalAveragePooling1D()(InS1)
    InS2 = InS_mechanism(64)([prot_LSTM, pept_LSTM])
    InS2_global_pool = GlobalAveragePooling1D()(InS2)

    # Concatenate & MLP
    joint_rep = Concatenate(axis=-1)([pept_MSA_global_pool, InS1_global_pool, InS2_global_pool, prot_MSA_global_pool])
    concat_Des = Dense(256)(joint_rep)
    concat_act = LeakyReLU(alpha = 0.02)(concat_Des)
    concat_Dro = Dropout(0.25)(concat_act)

    concat_Des = Dense(128)(concat_Dro)
    concat_Dro = LeakyReLU(alpha = 0.02)(concat_Des)
    output = Dense(2, activation='softmax')(concat_Dro)
    
    model = Model(inputs=[pept_emb_input, prot_mat_input],outputs= output)
    model.summary()
    
    return model
