# Video Surveillance for Road Traffic Monitoring
Team 5

## Contributors

| Dani Azemar | Richard Segovia |  Sergi Solà |   Sergio Casas  |
|-------------|-----------------|-------------|------------------|
|  hamddan4   |   richardseba   | sergiisolaa | sergiocasaspastor|

[Final presentation slides](https://docs.google.com/presentation/d/1pDFVmdLl78mPiCXpMbp4RYjThJxFH2t5P364LMznqPg/edit?usp=sharing)

## Project description

This project aims to implement a system able to perform multi-object tracking in both single or multiple cameras.

On week 1 we implemented the main metrics used for the evaluation of the predicted detections and optical flow estimations. 

On week 2 we developed some methods to perform foreground-background segmentation such as gaussian distribution or recursive gaussian modelling. 

Then, on week 3, we fine-tuned some object detectors from the Detectron2 framework to perform car detection, implemented the maximum overlap method for object tracking and introduced also the Kalman filter to improve its performance. 

On week 4 we implemented a simple method for optical flow estimation and tested some of the state-of-the-art methods for that purpose. Then we implemented a method for video stabilization based on our method for estimating optical flow and compared it to the state-of-the-art and finally we tried to improve the multi-object tracking for single camera using the optical flow estimations. 

Finally, on week 5, we used all the previous work done on the project to implement a system for Multi-object Tracking on Single Camera. Then, using that method we implemented a system for Multi-object Tracking on Multiple Cameras using a triplet network and a complex algorithm based on the euclidean distances between the embeddings obtained from the triplet network and a voting scheme based on the corresponding minimums on that distance for the different frames of a track to relate tracklets between cameras. 

## Week 1

### Tasks

- Task 1: Detection metrics.
- Task 2: Detection metrics. Temporal analysis.
- Task 3: Optical flow evaluation metrics.
- Task 4: Visual representation optical flow.

### Execution
Copy the dataset folder into the same folder as this repository.

Execute tasks 1 and 2:
```
python src/main.py
```

Execute tasks 3 and 4:
```
python src/opflows/test.py
```

## Week 2

### Tasks

- Task 1: Gaussian distribution
- Task 2: Recursive Gaussian modeling
- Task 3: Compare with state-of-the-art
- Task 4: Color sequences

### Execution
Copy the dataset folder into the same folder as this repository.

Execute task1:
```
python src/main.py --general_config detector.alpha=4 detector.rho=0

```

Execute task 2:
```
python src/main.py --general_config detector.alpha=5 detector.rho=0.01
```
Gridsearch
Change parameters of grid search inside grid_search.py
```
python src/grid_search.py
```

Execute task 3:
```
python src/main.py --general_config detector.type="gauss_black_rem"
```
Detectors: ["color_gauss_black_rem","gauss_black_rem", "MOG", "MOG2", "CNT", "GMG", "LSBP", "GSOC"]

Execute task 4:
Change Color space inside main.py file
```
python src/main.py 
```
## Week 3

### Tasks

- Task 1: Object detection
    - Task 1.1: Object Detection: Off-the-Shelf
    - Task 1.2: Object Detection: Fine-tuned model
    
- Task 2: Object tracking
    - Task 2.1: Tracking by maximum overlap
    - Task 2.2: Tracking with a Kalman filter
    - Task 2.3: IDF1 for Multiple Object Tracking
 
 ### Execution
 Copy the dataset folder into the same folder as this repository.
 
All the tasks can be executed using the following command:
```
python src/main.py 
```
The parameters that can be modified can be found on the src/config/general.py file. The values for that parameters can be modified using a command in the terminal. For example, to run a task with a different value for the parameter weights_path we can use:
```
python src/main.py --general_config detector.weights_path="../new_folder/detectron.weights"
```
## Week 4

### Tasks

- Task 1: Optical Flow
    - Task 1.1: Optical Flow with Block Matching
    - Task 1.2: Off-the-shelf Optical Flow
- Task 2: Video Stabilization
    - Task 2.1: Video Stabilization with Block Matching
    - Task 2.2: Off-the-shelf Video Stabilization
- Task 3: Object Tracking
    - Task 3.1: Object Tracking with Optical Flow

### Execution
Copy the dataset into the same folder as this repository.

#### Task 1

All the methods used in task 1 can be tested by changing the sel_method index to the one of the desired method from the src/opflows/test.py file. Then we can just execute the test.py file. 

#### Task 2

Our method for video stabilization, that uses Block Matching, can be executed using the following command:

```
python src/opflows/block_matching.py
```

The first off-the-shelf method for video stabilization, the one based on OpenCV, can be executed using the following command:

```
python src/scripts/video_stabilizer.py
```

However, it might be interesting to enter the file and change the path to the videos we are working on.

The other off-the-shelf method for video stabilization, the Kamran Video Stabilizer, is implemented on Matlab and need to be executed from there. It can be executed using the main.m file located on src/scripts/Kamran_video_stabilizer/main.m.

#### Task 3

To execute task 3 you can just execute the main.py file or just execute the following command on the terminal:

```
python src/main.py --general_config tracker.ttype = "optical_flow_track"
```

## Week 5
### Tasks
- Task 1: Multi-object Tracking with Single Camera
- Task 2: Multi-object Tracking with Multiple Cameras

### Execution
#### Task 1
The model for MTSC (Task 1) can be executed by using the following command on the terminal:
```
python src/MTSC.py --general_config tracker.ttype = "optical_flow_track"
```
The parameter tracker.ttype can take the following values: "overlap", "centroid", "sort" and "optical_flow_track". All the other modifiable parameters with its possible values can be found on config/general.py. Moreover, inside MTSC.py, you can find two flags (train and test). You can set train to True if you want to train a new model for MTSC. Set the test flag to True if you want to test an existing model. Inside MTSC/test_detectron.py and MTSC/train_detectron.py there are more parameters you might need to set. Important: check that the paths to the dataset are correct.

#### Task 2
The system for MTMC (Task 2) can be executed by using the following command on the terminal. However, to run that system we need to have the results obtained from the system from Task 1:
```
python src/MTMC/test.py 
```
To train the triplet network, you should check [the readme file](https://github.com/mcv-m6-video/mcv-m6-2020-team5/blob/master/src/MTMC/reid_training/README.md) for the network.

All the modifiable parameters for that system(cameras, feature normalization, load_pickles, max_permitted_size,...) can be found at the beginning of the main function from that file. Important: check that the paths to the dataset are correct.


 
