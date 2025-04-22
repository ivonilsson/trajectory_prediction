import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        embed_mlp=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.embed_mlp = embed_mlp
        if self.embed_mlp:
            self.mlp = keras.Sequential([
                layers.Dense(units,
                            activation="relu",
                            kernel_initializer=self.kernel_initializer,
                            kernel_regularizer=self.kernel_regularizer),
                layers.Dense(units,
                            kernel_initializer=self.kernel_initializer,
                            kernel_regularizer=self.kernel_regularizer),
            ])

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="kernel_attention",
        )
        super().build(input_shape)

    def call(self, inputs):
        node_states, edges = inputs
        if self.embed_mlp:
            node_states_transformed = self.mlp(node_states)
        else:
            node_states_transformed = tf.matmul(node_states, self.kernel)
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        # normalize without repeat
        scores_sum = tf.math.unsorted_segment_sum(
            attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0]
        )
        norm_den = tf.gather(scores_sum, edges[:, 0])
        attention_norm = attention_scores / norm_den
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out

class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", embed_mlp=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.embed_mlp = embed_mlp
        self.attention_layers = [GraphAttention(units, embed_mlp=embed_mlp) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs
        outputs = [
            att([atom_features, pair_indices])
            for att in self.attention_layers
        ]
        if self.merge_type == "concat":
            x = tf.concat(outputs, axis=-1)
        else:
            x = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        return tf.nn.relu(x)

class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        embed_mlp=False,
        top_layer_mlp=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # original tutorial names
        self.node_states = node_states
        self.edges = edges
        self.embed_mlp = embed_mlp
        if top_layer_mlp:
            self.preprocess = keras.Sequential([
            layers.Dense(hidden_units * num_heads, activation="relu"),
            layers.Dense(hidden_units * num_heads),
            ])
        else:
            self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads, embed_mlp=embed_mlp) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        return self.output_layer(x)

    def train_step(self, data):
        indices, labels = data
        with tf.GradientTape() as tape:
            outputs = self([self.node_states, self.edges], training=True)
            preds = tf.gather(outputs, indices)
            loss = self.compiled_loss(labels, preds)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(labels, preds)
        x = {m.name: m.result() for m in self.metrics}
        x["loss"] = loss
        return x

    def test_step(self, data):
        indices, labels = data
        outputs = self([self.node_states, self.edges], training=False)
        preds = tf.gather(outputs, indices)
        loss = self.compiled_loss(labels, preds)
        self.compiled_metrics.update_state(labels, preds)
        x = {m.name: m.result() for m in self.metrics}
        x["loss"] = loss
        return x