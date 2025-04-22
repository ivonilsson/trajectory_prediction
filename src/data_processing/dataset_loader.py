import glob
import os
import pandas as pd
import numpy as np

def load_and_format_data(data_dir, undirected=False):
    node_files = sorted(glob.glob(os.path.join(data_dir, "*.nodes")))
    edge_files = sorted(glob.glob(os.path.join(data_dir, "*.edges")))
    if not node_files:
        raise FileNotFoundError(f"No .nodes files in {data_dir}")
    if not edge_files:
        raise FileNotFoundError(f"No .edges files in {data_dir}")

    nodes_list, edges_list = [], []
    for scene_id, (nfile, efile) in enumerate(zip(node_files, edge_files)):
        # --- Nodes ---
        cols = ["node_id","x_now","y_now","x_prev","y_prev","x_next","y_next"]
        df_nodes = pd.read_csv(nfile, header=None, names=cols, na_values=["_"])
        df_nodes.dropna(subset=cols, inplace=True)
        df_nodes = df_nodes.reset_index().rename(columns={"index":"original_index"})
        df_nodes['scene_id'] = scene_id
        # compute velocity & angles
        df_nodes['vel_x'] = df_nodes['x_now'] - df_nodes['x_prev']
        df_nodes['vel_y'] = df_nodes['y_now'] - df_nodes['y_prev']
        #df_nodes['vel'] = np.hypot(df_nodes['vel_x'], df_nodes['vel_y'])
        df_nodes['angle_motion'] = np.arctan2(df_nodes['vel_y'], df_nodes['vel_x'])
        nodes_list.append(df_nodes)
        # --- Edges ---
        df_edges = pd.read_csv(efile, header=None, names=["target_id","source_id"])
        df_edges = df_edges[(df_edges.target_id >= 0) & (df_edges.source_id >= 0)].copy()
        mapping = dict(zip(df_nodes['node_id'], df_nodes['original_index']))
        df_edges['scene_id'] = scene_id
        df_edges['target_idx'] = df_edges['target_id'].map(mapping)
        df_edges['source_idx'] = df_edges['source_id'].map(mapping)
        df_edges.dropna(subset=["target_idx","source_idx"], inplace=True)
        df_edges[['target_idx','source_idx']] = df_edges[['target_idx','source_idx']].astype(int)
        edges_list.append(df_edges[['scene_id','target_idx','source_idx']])

    # concatenate scenes
    nodes_df = pd.concat(nodes_list, ignore_index=True)
    edges_df = pd.concat(edges_list, ignore_index=True)

    # assign global unique index
    nodes_df = nodes_df.reset_index().rename(columns={'index':'global_idx'})
    # map scene-local indices to global_idx
    edges_df = edges_df.merge(
        nodes_df[['scene_id','original_index','global_idx']],
        left_on=['scene_id','target_idx'],
        right_on=['scene_id','original_index'],
        how='inner'
    ).rename(columns={'global_idx':'target'})
    edges_df = edges_df.merge(
        nodes_df[['scene_id','original_index','global_idx']],
        left_on=['scene_id','source_idx'],
        right_on=['scene_id','original_index'],
        how='inner', suffixes=('','_src')
    ).rename(columns={'global_idx':'source'})
    edges_df = edges_df[['target','source']].astype(int)
    if undirected:
        rev = edges_df.rename(columns={'target':'source','source':'target'})
        edges_df = pd.concat([edges_df, rev], ignore_index=True)
    return nodes_df, edges_df