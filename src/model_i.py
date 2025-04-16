import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)
np.random.seed(2)

class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        
        #replaced commented section below
        #attention_scores_sum = tf.repeat(
        #    attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        #)
        #attention_scores_norm = attention_scores / attention_scores_sum

        #With this to avoid XLA issues
        scores_sum_for_each_edge = tf.gather(attention_scores_sum, edges[:, 0])

        # Normalize
        attention_scores_norm = attention_scores / scores_sum_for_each_edge

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)

class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs


def train_model_on_separate_scenes(model, train_scenes, epochs=10, learning_rate=1e-3):
    """
    Custom training loop over multiple scenes for regression on next_x, next_y
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        scene_losses = []
        for scene in train_scenes:
            #convert data to tensors
            node_states = tf.convert_to_tensor(scene["features"], dtype=tf.float32)
            edges = tf.convert_to_tensor(scene["edges"], dtype=tf.int32)
            targets = tf.convert_to_tensor(scene["targets"], dtype=tf.float32)

            #optionally shuffle node indices, or just take them all
            indices = tf.range(tf.shape(node_states)[0])

            with tf.GradientTape() as tape:
                outputs = model((node_states, edges))  #shape [N, 2]
                preds = tf.gather(outputs, indices)    #shape [N, 2]
                y_true = tf.gather(targets, indices)   #shape [N, 2]
                loss = loss_fn(y_true, preds)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            scene_losses.append(loss.numpy())

        print(f"Epoch {epoch+1}/{epochs} | avg scene loss: {np.mean(scene_losses):.4f}")


def evaluate_on_separate_scenes(model, test_scenes):
    """
    Compute the average Euclidean distance between predicted and true positions across all test scenes
    """
    distances = []
    for scene in test_scenes:
        node_states = tf.convert_to_tensor(scene["features"], dtype=tf.float32)
        edges = tf.convert_to_tensor(scene["edges"], dtype=tf.int32)
        targets = tf.convert_to_tensor(scene["targets"], dtype=tf.float32)

        outputs = model((node_states, edges))  #shape [N, 2]
        diff = outputs - targets
        dist = tf.norm(diff, axis=1)  #L2 norm => shape [N]
        distances.extend(dist.numpy())

    mean_dist = np.mean(distances)
    #data is in millimeters
    print(f"Average Euclidean distance error (on test data) = {mean_dist/1000:.2f} meters")
    return mean_dist