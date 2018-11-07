from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.layers import Activation, concatenate, add, multiply
from model.attention_decoder import AttentionDecoder

class seq2seq_attention:
    def __init__(self, num_encoder_tokens, embedding_dim,
                 hidden_dim, num_decoder_tokens):
        self.num_encoder_tokens = num_encoder_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_decoder_tokens = num_decoder_tokens

    def get_model(self):
        # Input text
        encoder_inputs = Input(shape=(None,))
        # word embedding layer
        encoder_inputs_emb = Embedding(input_dim=self.num_encoder_tokens,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True)(encoder_inputs)

        rnn_encoded = Bidirectional(LSTM(self.hidden_dim, return_sequences=True),
                                    merge_mode='concat')(encoder_inputs_emb)

        y_hat = AttentionDecoder(self.hidden_dim,
                                 name='attention_decoder_1',
                                 output_dim=n_labels,
                                 return_probabilities=return_probabilities,
                                 trainable=trainable)(rnn_encoded)

        model = Model(inputs=input_, outputs=y_hat)

        return model
