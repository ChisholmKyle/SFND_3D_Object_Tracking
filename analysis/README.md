# Data Analysis for 3D Tracking

Thirty five different feature extraction test cases were performed on the sample sequence of images with data written to a CSV file. Sample images were generated showing lidar top view with bounding box and camera view with lidar and keypoint extraction overlay for the center lane vehicle. The processing time and time to collision (TTC) values were recorded for each test case.

The output plots can be generated with new data produced by running `3D_feature_tracking` and following the instructions in the last section of this report.

## TTC Performance Evaluation

### FP.5 Lidar Evaluation

### FP.6 Camera TTC Estimation

![Plot of image vs. TTC](data/results_2020-04-19_12h04m44s/test_data_ttc.png)

![2D mesh plot of duration](data/results_2020-04-19_12h04m44s/test_data_duration_mesh.png)

## How to Generate Plots From Csv

Create virtual environment and install prerequisites

```bash
python3 -m venv .venv
source ./.venv/bin/activate
python -m pip install numpy matplotlib
```

In file [process3DTrackingData.py](process3DTrackingData.py), modify the following lines to use a different csv file and affect other output parameters:

```python
# input data
data_file = 'analysis/data/results_2020-04-19_12h04m44s/test_data.csv'

# show plots
display_plots = True

# output plot size
plot_size = {
    'width': 7.5,
    'height': 5,
    'dpi': 96
}

# table output size
table_size = {
    'width': 6,
    'height': 9,
    'dpi': 96
}
```

Then run script:

```bash
python ./analysis/process2DfeatureData.py
```
