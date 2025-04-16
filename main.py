import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.datamodule import load_data
from src.model_i import GraphAttentionNetwork, train_model_on_separate_scenes, evaluate_on_separate_scenes

def main():
        #dm.pre_process(path="data/dataset")
    np.random.seed(2)
    scenes = load_data()

    num_scenes = len(scenes)
    indices = np.arange(num_scenes)
    np.random.shuffle(indices)

    train_scenes_idx = indices[: int(0.8 * num_scenes)]
    test_scenes_idx  = indices[int(0.8 * num_scenes):]

    train_scenes = [scenes[i] for i in train_scenes_idx]
    test_scenes  = [scenes[i] for i in test_scenes_idx]

    print(f"Train scenes: {len(train_scenes)}, Test scenes: {len(test_scenes)}")

    # Define hyper-parameters
    HIDDEN_UNITS = 64 # 100
    NUM_HEADS = 8
    NUM_LAYERS = 3
    OUTPUT_DIM = 2

    #not using some of these as we have custom training and test eval for separate scenes, also using adam instead of scg with momentum
    NUM_EPOCHS = 100
    #BATCH_SIZE = 256
    #VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 1e-3 #3e-1 #maybe change
    #MOMENTUM = 0.9

    gat_model = GraphAttentionNetwork(
        hidden_units=HIDDEN_UNITS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM
    )

    train_model_on_separate_scenes(gat_model, train_scenes, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    evaluate_on_separate_scenes(gat_model, test_scenes)




if __name__ == "__main__":
    main()
