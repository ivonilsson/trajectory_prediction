import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.model.model import GraphAttentionNetwork
from src.data_processing.dataset_loader import load_and_format_data
from src.vis.plots import plot_random_test_sub_sample
from src.util.utils import save_results

def main():
    # prep
    print("Running prep work...")
    with open("config/cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    DATA_DIR      = cfg["data_dir"]
    EPOCHS        = cfg["epochs"]
    LR            = cfg["learning_rate"]
    BATCH_SIZE    = cfg["batch_size"]
    SCALED        = cfg["scaled"]

    HIDDEN_UNITS  = cfg["hidden_units"]
    NUM_HEADS     = cfg["num_heads"]
    NUM_LAYERS    = cfg["num_layers"]

    SEED          = cfg["seed"]
    PLOT_ROWS     = cfg["num_plot_rows"]
    PLOT_COLS     = cfg["num_plot_cols"]

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)

    # task 1 below
    nodes_df, edges_df = load_and_format_data(DATA_DIR)
    print(f"Nodes: {nodes_df.shape}, Edges: {edges_df.shape}")

    features = ['x_now','y_now','x_prev','y_prev','vel_x','vel_y','angle_motion']
    targets = ['x_next', 'y_next']

    X = nodes_df[features].to_numpy(np.float32)
    Y = nodes_df[targets].to_numpy(np.float32)
    edges_idx = edges_df.to_numpy(np.int32)

    N = len(X)
    perm = np.random.permutation(N)
    n_train = int(0.7*N)
    n_val = int(0.15*N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train+n_val]
    test_idx  = perm[n_train+n_val:]

    if SCALED:
        X_tr = X[train_idx]
        mins_X, maxs_X = X_tr.min(0), X_tr.max(0)
        X = (X - mins_X)/(maxs_X-mins_X+1e-6)
        Y_tr = Y[train_idx]
        mins_Y, maxs_Y = Y_tr.min(0), Y_tr.max(0)
        Y = (Y - mins_Y)/(maxs_Y-mins_Y+1e-6)
    else:
        X, Y = X, Y

    train_ds = tf.data.Dataset.from_tensor_slices((train_idx, Y[train_idx])).batch(BATCH_SIZE)
    val_ds   = tf.data.Dataset.from_tensor_slices((val_idx,   Y[val_idx]  )).batch(BATCH_SIZE)
    _  = tf.data.Dataset.from_tensor_slices((test_idx,  Y[test_idx] )).batch(BATCH_SIZE)

    node_states  = tf.convert_to_tensor(X, tf.float32)
    edges_tensor = tf.convert_to_tensor(edges_idx, tf.int32)

    print("Prep completed.")

    print("Running Task 1...")

    model = GraphAttentionNetwork(
        node_states=node_states,
        edges=edges_tensor,
        hidden_units=HIDDEN_UNITS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        output_dim=Y.shape[1],
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR),
        loss='mse',
        metrics=[keras.metrics.MeanAbsoluteError(name='mae')]
    )
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[earlystop_cb], verbose=2)

    label = "Scaled" if SCALED else "Raw"

    outputs_all = model([node_states, edges_tensor], training=False).numpy()
    y_pred = outputs_all[test_idx]
    y_true = Y[test_idx]

    if SCALED:
        preds_real = y_pred*(maxs_Y-mins_Y+1e-6)+mins_Y
        truth_real = y_true*(maxs_Y-mins_Y+1e-6)+mins_Y
    else:
        preds_real, truth_real = y_pred, y_true

    mse  = np.mean((preds_real - truth_real)**2)
    mae  = np.mean(np.abs(preds_real - truth_real))
    eucl = np.mean(np.linalg.norm(preds_real - truth_real, axis=1))

    results = [
        {
            "Label": label,
            "MSE": f"{mse:.4f}",
            "MAE": f"{mae:.4f}",
            "Euclidean": f"{eucl:.4f}"
        }
    ]
    save_results("results/task_1.txt", results)

    plot_random_test_sub_sample(nodes_df, test_idx, preds_real, truth_real,num_rows=PLOT_ROWS, num_cols=PLOT_COLS, seed=SEED, save_dir="plots/task_1/task_1.png")
    print(f"Task 1 completed. Saved results and plot(s).")

    # task 2 below
    print("Running Task 2...")
    NUM_HEADS     = cfg["task_2"]["num_heads"]
    EMBED_MLP     = cfg["task_2"]["embed_mlp"]
    #TOP_LAYER_MLP = cfg["task_2"]["top_layer_mlp"]

    for num_heads in NUM_HEADS:
        print(f"Initializing GAN with {num_heads} number of attention heads.")
        model = GraphAttentionNetwork(
            node_states=node_states,
            edges=edges_tensor,
            hidden_units=HIDDEN_UNITS,
            num_heads=num_heads,
            num_layers=NUM_LAYERS,
            output_dim=Y.shape[1],
            embed_mlp=EMBED_MLP,
            #top_layer_mlp=TOP_LAYER_MLP
        )
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=LR),
            loss='mse',
            metrics=[keras.metrics.MeanAbsoluteError(name='mae')]
        )
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

        label = "Scaled" if SCALED else "Raw"

        outputs_all = model([node_states, edges_tensor], training=False).numpy()
        y_pred = outputs_all[test_idx]
        y_true = Y[test_idx]

        if SCALED:
            preds_real = y_pred*(maxs_Y-mins_Y+1e-6)+mins_Y
            truth_real = y_true*(maxs_Y-mins_Y+1e-6)+mins_Y
        else:
            preds_real, truth_real = y_pred, y_true

        mse  = np.mean((preds_real - truth_real)**2)
        mae  = np.mean(np.abs(preds_real - truth_real))
        eucl = np.mean(np.linalg.norm(preds_real - truth_real, axis=1))

        results = [
            {
                "Label": label,
                "MSE": f"{mse:.4f}",
                "MAE": f"{mae:.4f}",
                "Euclidean": f"{eucl:.4f}"
            }
        ]
        save_path = f"task_2_num_heads_{num_heads}"
        save_results(f"results/{save_path}.txt", results)

        plot_random_test_sub_sample(nodes_df, test_idx, preds_real, truth_real,num_rows=PLOT_ROWS, num_cols=PLOT_COLS, seed=SEED, save_dir=f"plots/task_2/{save_path}.png")
        print(f"Task 2 completed. Saved results and plot(s).")

    # task 3 below
    print("Running Task 3...")
    COSINE = cfg["task_3"]["cosine"]

    model = GraphAttentionNetwork(
        node_states=node_states,
        edges=edges_tensor,
        hidden_units=HIDDEN_UNITS,
        num_heads=num_heads,
        num_layers=NUM_LAYERS,
        output_dim=Y.shape[1],
        cosine=COSINE
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR),
        loss='mse',
        metrics=[keras.metrics.MeanAbsoluteError(name='mae')]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    label = "Scaled" if SCALED else "Raw"
    outputs_all = model([node_states, edges_tensor], training=False).numpy()
    y_pred = outputs_all[test_idx]
    y_true = Y[test_idx]

    if SCALED:
        preds_real = y_pred*(maxs_Y-mins_Y+1e-6)+mins_Y
        truth_real = y_true*(maxs_Y-mins_Y+1e-6)+mins_Y
    else:
        preds_real, truth_real = y_pred, y_true

    mse  = np.mean((preds_real - truth_real)**2)
    mae  = np.mean(np.abs(preds_real - truth_real))
    eucl = np.mean(np.linalg.norm(preds_real - truth_real, axis=1))

    results = [
        {
            "Label": label,
            "MSE": f"{mse:.4f}",
            "MAE": f"{mae:.4f}",
            "Euclidean": f"{eucl:.4f}"
        }
    ]
    save_results(f"results/task_3.txt", results)

    plot_random_test_sub_sample(nodes_df, test_idx, preds_real, truth_real,num_rows=PLOT_ROWS, num_cols=PLOT_COLS, seed=SEED, save_dir="plots/task_3/task_3.png")
    print(f"Task 3 completed. Saved results and plot(s).")

if __name__ == '__main__':
    main()