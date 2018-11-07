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
        # LSTM encoder layer
        # We discard `encoder_outputs` and only keep the states.
        _, encoder_h, encoder_c = LSTM(self.hidden_dim,
                                       return_state=True,
                                       return_sequences=False)(encoder_inputs_emb)
        encoder_states = [encoder_h, encoder_c]

        # decoder input: 'encoder_states' & summary
        decoder_inputs = Input(shape=(None,))
        # word embedding layer
        decoder_inputs_emb = Embedding(input_dim=self.num_decoder_tokens,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True)(decoder_inputs)

        # LSTM encoder layer
        # We set up our decoder to return full output sequences,
        decoder_outputs, decoder_h, encoder_c = LSTM(self.hidden_dim,
                                                     return_sequences=True,
                                                     return_state=True)(decoder_inputs_emb,
                                                                        initial_state=encoder_states)
        # Attention
        # encoder_outputs: (BxTxH)
        # decoder_outputs: (BxTxH)
        # W_h * h_i
        wh = Dense(self.hidden_dim, use_bias=False)(encoder_outputs)
        # W_s * s_t + bias
        ws = Dense(self.hidden_dim, use_bias=True)(decoder_outputs)
        comb = add([wh, ws])
        comb = Activation('tanh')(comb)
        attention = Dense(self.hidden_dim, use_bias=False, activation='softmax')(comb)
        # h*: context
        context = multiply([attention, encoder_outputs])
        decoder_combined_context = concatenate([context, decoder_outputs])
        output = Dense(self.hidden_dim)(decoder_combined_context)
        output = Dense(self.num_decoder_tokens, activation="softmax")(output)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], output)

        return model
