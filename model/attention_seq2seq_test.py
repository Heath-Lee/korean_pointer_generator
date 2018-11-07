from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional
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
        # LSTM encoder
        encoder_o, encoder_h, encoder_c = Bidirectional(LSTM(self.hidden_dim,
                                                             return_sequences=True,
                                                             return_state=True),
                                                        merge_mode='concat')(encoder_inputs_emb)

        # Input summaru
        decoder_inputs = Input(shape=(None,))
        # word embedding layer
        decoder_inputs_emb = Embedding(input_dim=self.num_decoder_tokens,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True)(decoder_inputs)
        # LSTM decoder
        decoder_o, decoder_h, decoder_c = LSTM(self.hidden_dim,
                                               return_sequences=True,
                                               return_state=True)(decoder_inputs_emb,
                                                                  initial_state=[encoder_h, encoder_c])
        # LSTM decoder + attention
        ''' 
        [o, context], h, c = LSTMdecoder(abstracted text, input summary, encoder states)            
        '''


        # output layers
        decoder_with_context = concatenate([decoder_o, context])
        h = Dense(self.hidden_dim)(decoder_with_context)
        decoder_outputs = Dense(self.num_decoder_tokens, activation='softmax')(h)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model
