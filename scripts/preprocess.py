import os
import numpy as np
import binvox_rw

def voxelize_data(input_dir, output_file):
    CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    X, y = {'train': [], 'test': []}, {'train': [], 'test': []}

    for label, cl in enumerate(CLASSES):
        for split in ['train', 'test']:
            examples_dir = os.path.join(input_dir, cl, split)
            if not os.path.exists(examples_dir):
                print(f"Warning: Directory {examples_dir} does not exist. Skipping...")
                continue
            for example in os.listdir(examples_dir):
                if 'binvox' in example:
                    with open(os.path.join(examples_dir, example), 'rb') as file:
                        data = np.int32(binvox_rw.read_as_3d_array(file).data)
                        padded_data = np.pad(data, 3, 'constant')
                        X[split].append(padded_data)
                        y[split].append(label)

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savez_compressed(output_file, X_train=X['train'], X_test=X['test'], y_train=y['train'], y_test=y['test'])

# Example usage:
if __name__ == '__main__':
    # Adjust the input directory to the absolute path of the ModelNet10 folder
    # voxelize_data('../ModelNet10/ModelNet10', '../data/modelnet10.npz')
    #incase training on gg colab
    voxelize_data('/content/drive/MyDrive/convert_3d_off_file/ModelNet10/ModelNet10', '/content/drive/MyDrive/convert_3d_off_file/data/modelnet10.npz')
