# Training ResNet50 with [TorchReID](https://github.com/KaiyangZhou/deep-person-reid) framework

In this folder you will find the scripts that we used to train a ResNet50 to perform a ReID task. For installation of the necessary libraries, follow the instructions on their repository.

## Scripts to execute
### create_dataset.py
Generates the folders dataset with the format of  a [common reID dataset](https://kaiyangzhou.github.io/deep-person-reid/datasets.html?highlight=datasets#market1501-dagger-market1501) for an easy training. To generate your own dataset you must change `input_dir` and `output_dir` with the reference to the train folder containing the necessary scenes, and then, the test with other desired scenes.

### [optional] reduce_dataset.py
This script will reduce a dataset. Needed if resulting bbounding boxes are too much for loading into memory.

### model_config.py
Contains the configuration of the model (resnet50) ready for training using the AI2019 dataset envelope. Before running the script, change `DATASET_PATH` to the real path of AI2019_Dataset. This folder of the dataset must contain folders `bounding_box_train` and `bounding_box_test_reduced` (you can change the train and test folders in `AI2019_dataset.py` file.

Running the script will cause:

 - Download the configuration of the model
 - Download the weights.
 - Load the AIDataset
 - Train

## Others
### AI2019_dataset.py
Creates the DataManager specific for the AI2019 Dataset. 

