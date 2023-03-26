from transformers import BertTokenizer, TFBertModel

import logging
import time

import numpy as np
import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        }
        return config


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class ConstraintMaskModule(tf.keras.Model):
    def __init__(self, obs_size, obs_channels, description_size, conv_filter, kernel_size, num_layers,
                 d_model,
                 num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.obs_size = obs_size
        self.description_size = description_size
        self.conv_filter = conv_filter
        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.encoder = Encoder(num_layers=num_layers,
        #                        d_model=d_model,
        #                        num_heads=num_heads,
        #                        dff=dff,
        #                        vocab_size=vocab_size,
        #                        dropout_rate=dropout_rate)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bertModel = TFBertModel.from_pretrained('bert-base-cased', return_dict=True)
        self.bertModel.trainable = False
        self.dense = tf.keras.layers.Dense(1, activation='relu')
        self.conv = tf.keras.layers.Conv2D(self.conv_filter,
                                           self.kernel_size,
                                           activation='relu',
                                           input_shape=(obs_size, obs_size, description_size + obs_channels))
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(obs_size * obs_size)
        ])
        # self.disc = tf.keras.layers.Discretization(bin_boundaries=[0.])

    @staticmethod
    def binary_activation(x):
        cond = tf.less(x, tf.zeros(tf.shape(x)))
        out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
        return out

    def call(self, inputs):
        input_ids, token_type_ids, attention_mask, o = inputs
        x = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        x = self.bertModel(x, training=False).last_hidden_state
        # x = self.encoder(x, training=False)
        x = self.dense(x)
        x = tf.squeeze(x, axis=-1)
        x = tf.tile(x, [1, self.obs_size * self.obs_size])
        x = tf.reshape(x, [-1, self.obs_size, self.obs_size, self.description_size])
        x = tf.concat([x, o], -1)
        x = self.conv(x)
        x = self.seq(x)
        return x


class ConstraintThresholdModule(tf.keras.Model):
    def __init__(self, obs_size, obs_channels, description_size, conv_filter, kernel_size, num_layers,
                 d_model,
                 num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.description_size = description_size
        self.conv_filter = conv_filter
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bertModel = TFBertModel.from_pretrained('bert-base-cased', return_dict=True)
        self.bertModel.trainable = False
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        # self.disc = tf.keras.layers.Discretization(bin_boundaries=[0.])

    @staticmethod
    def binary_activation(x):
        cond = tf.less(x, tf.zeros(tf.shape(x)))
        out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
        return out

    def call(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs
        x = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        x = self.bertModel(x, training=False).last_hidden_state
        x = self.seq(x)
        return x


if __name__ == '__main__':
    # Instantiate the encoder.
    BATCH_SIZE = 128
    description_size = 100
    # model = ConstraintMaskModule(obs_size=7,
    #                              obs_channels=3,
    #                              description_size=10,
    #                              conv_filter=10,
    #                              kernel_size=3,
    #                              num_layers=4,
    #                              d_model=512,
    #                              num_heads=8,
    #                              dff=2048,
    #                              vocab_size=8500)

    model = ConstraintThresholdModule(obs_size=7,
                                      obs_channels=3,
                                      description_size=10,
                                      conv_filter=10,
                                      kernel_size=3,
                                      num_layers=4,
                                      d_model=512,
                                      num_heads=8,
                                      dff=2048,
                                      vocab_size=8500)

    # test_constraint = tf.zeros(
    #     (BATCH_SIZE, 10),
    #     dtype=tf.dtypes.float32,
    #     name='constraint'
    # )

    test_constraint = []
    for i in range(BATCH_SIZE):
        test_constraint.append("test")

    test_obs = tf.zeros(
        (BATCH_SIZE, 7, 7, 3),
        dtype=tf.dtypes.float32,
        name='obs'
    )

    test_constraint = ["test" for i in range(BATCH_SIZE)]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    inputs = tokenizer(test_constraint, return_tensors="tf", padding="max_length", max_length=description_size)

    # sample_encoder_output = model((inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], test_obs))
    sample_encoder_output = model((inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']))
    model.summary()

    # Print the shape.
    print(len(test_constraint))
    print(test_obs.shape)
    print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.
