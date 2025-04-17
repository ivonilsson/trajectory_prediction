import os
import pandas as pd

def pre_process(path='data/dataset'):
    """Preprocesses the data, dataset consists of two files for each traffic scene:
    
    For file:
    
    two columns containing node IDs
    target, source
    Note: The tutorial models directed edges with source -> target.
    You can either use undirected edges by changing the implementation or adding the missing entries to the edges file,
    e.g., to the line target, source, you add the line source target. If you want to be more fancy, you could also try to infer
    which other pedestrians the source node can see in their field of view and only add those (this would model that the movement
    decisions are based only on the pedestrians in the field of view.)

    <scene_id>.nodes

    seven columns with node properties and target values, which should be predicted 
    node id, current x, current y, previous x, previous y, future x, future y
    the previous x and y represents the location of the pedestrian 1 second ago (you can use those values directly or infer the
    movement direction and some speed estimate yourself)
    the future x and y represents the target value, i.e., the location where the pedestrian will be in 1 second
    Note: Some pedestrians do not have a future x and y coordinate, so you need to filter those for prediction. However, you can
    still use their current and previous location when predicting the future location of other pedestrians.
    """

    # THE IDEA HERE IS TO FIX THE DATA IN ALL THOSE FILES ABOVE SOMEHOW
    print('Starting image filtering.')
    num_skipped = 0
    f_names = os.listdir(path)
    f_names.sort()
    for f_name in f_names:
        print('FILE NAMES:',f_name) #debugging

def pre_process_full_graph(path='data/dataset', output_path='data/processed/'):
    """
    Creates one large graph for all datapoints
    """
    os.makedirs(output_path, exist_ok=True)

    # THE IDEA HERE IS TO FIX THE DATA IN ALL THOSE FILES ABOVE SOMEHOW
    edge_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.edges')]
    node_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.nodes')]

    #scene_ids = sorted(set(f.replace(".nodes", "") for f in os.listdir(path) if f.endswith(".nodes")))
    print(f"Found {len(edge_files)} edge files and {len(node_files)} node files.")

    df_edges = pd.concat(map(get_edges, edge_files))
    df_nodes = pd.concat(map(get_nodes, node_files))
    
    edges_out = os.path.join(output_path, "full_graph_edges.edges")
    nodes_out = os.path.join(output_path, "full_graph_nodes.nodes")

    df_edges.to_csv(edges_out, index=False)
    df_nodes.to_csv(nodes_out, index=False)



def load_data(path='data/dataset/'):
    """
    Loads in all the traffic scenes, joins togheter data for corresponding .nodes and .edges files.
    """
    print('Starting data loading...')
    traffic_scenes = []
    #scene_ids = sorted(os.listdir(path))
    #print(scene_ids)
    #scene_ids.sort()
    #print(scene_ids)
    scene_ids = sorted(set(f.replace(".nodes", "") for f in os.listdir(path) if f.endswith(".nodes")))
    for scene_id in scene_ids:
        #print('FILE NAMES:',f_name) #debugging
        edges = get_edges(path + scene_id + ".edges")
        nodes = get_nodes(path + scene_id + ".nodes")

        if edges is None or nodes is None:
            print(f"Skipping {scene_id} due to missing data.")
            continue

        #remove all instances of _, will use this for now, could look to make use of these instances further on
        mask_valid = ~nodes.isin(["_", None]).any(axis=1)
        nodes = nodes[mask_valid].copy()

        cols = ["curr_x", "curr_y", "prev_x", "prev_y", "next_x", "next_y"]
        nodes[cols] = nodes[cols].astype(float)
        #print(type(nodes[cols]))

        node_idx = {node_id: idx for idx, node_id in enumerate(nodes["node_id"])}
        edges = edges[edges["source"].isin(node_idx) & edges["target"].isin(node_idx)]

        #make bidirectional
        reversed_edges = edges.rename(columns={"source": "target", "target": "source"})
        edges = pd.concat([edges, reversed_edges], ignore_index=True)

        edge_indices = edges[["target", "source"]].applymap(node_idx.get).to_numpy()

        if len(edge_indices) == 0:
            print(f"Skipping {scene_id} because it has 0 edges after filtering.")
            continue

        features = nodes[["curr_x", "curr_y", "prev_x", "prev_y"]].to_numpy()
        
        #targets: next positions (to predict)
        targets = nodes[["next_x", "next_y"]].to_numpy()

        traffic_scenes.append({
            "scene_id": scene_id,
            "features": features,
            "targets": targets,
            "edges": edge_indices,
        })
    return traffic_scenes

def get_edges(f_name):
    if '.edges' in f_name:
        print("Extracting edges from:", f_name)
        edges = pd.read_csv(
        f_name ,
        sep=",",
        header=None,
        names=["target", "source"],
        )
        return edges
    else:
        return None

def get_nodes(f_name):
    if '.nodes' in f_name:
        nodes = pd.read_csv(
                f_name,
                sep=",",
                header=None,
                names=["node_id", "curr_x", "curr_y", "prev_x", "prev_y", "next_x", "next_y"]
            )
        return nodes
    else:
        return None
