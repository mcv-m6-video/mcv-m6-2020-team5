# Video Surveillance for Road Traffic Monitoring
Team 5

## Contributors

| Dani Azemar | Richard Segovia |  Sergi Solà |   Sergio Casasa  |
|-------------|-----------------|-------------|------------------|
|  hamddan4   |   richardseba   | sergiisolaa | sergiocasaspastor|

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
 
 All the tasks can be executed 
 
