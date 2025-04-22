import os
import matplotlib.pyplot as plt
import numpy as np

def plot_random_test_sub_sample(nodes_df, test_idx, preds_real, truth_real, num_rows=3, num_cols=3, seed=42, save_dir=None):
    np.random.seed(seed)
    n_samples = num_rows * num_cols
    sel = np.random.choice(len(test_idx), size=n_samples, replace=False)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    axes = axes.flatten()

    for ax, i in zip(axes, sel):
        gid = test_idx[i]
        # positions
        x_prev, y_prev = nodes_df.loc[gid, ['x_prev','y_prev']]
        x_now,  y_now  = nodes_df.loc[gid, ['x_now','y_now']]
        x_true, y_true = truth_real[i]
        x_pred, y_pred = preds_real[i]

        # True trajectory: prev -> now -> true
        ax.plot([x_prev, x_now, x_true],
                [y_prev, y_now, y_true],
                'o-', label='True', zorder=1, color='blue')

        # Prediction: now -> pred
        ax.plot([x_now, x_pred],
                [y_now, y_pred],
                'o-', label='Pred', zorder=0, color='orange')

        # Start point on top
        ax.scatter(x_prev, y_prev,
                   color='red', marker='o', s=60,
                   label='Start', zorder=2)

        # Center viewport around current (x_now, y_now)
        all_x = [x_prev, x_now, x_true, x_pred]
        all_y = [y_prev, y_now, y_true, y_pred]
        span = max(max(all_x)-min(all_x), max(all_y)-min(all_y)) * 1.2 + 1e-6
        ax.set_xlim(x_now - span, x_now + span)
        ax.set_ylim(y_now - span, y_now + span)

        ax.set_aspect('equal', 'box')
        ax.set_title(f"Sample idx {gid}")
        ax.legend(fontsize='small')

    fig.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        out_dir = os.path.dirname(save_dir)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_dir)
