
import os
import matplotlib.pyplot as plt
import numpy as np

# %%

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

# time between images
step_size = 1 / 10

# plot only these case (some show terribale results, and too many cases to visualize)
test_cases_to_plot = {
    'FAST': {
        'BRISK', 'BRIEF', 'ORB', 'SIFT'
    },
    'BRISK': {
        'BRIEF', 'ORB'
    }
}

# %%
# Load data
data_type = {'names': ('case', 'image',
                       'detector', 'descriptor',
                       'camera_ttc', 'lidar_ttc',
                       'processing_time'),
             'formats': ('i', 'i',
                         'U10', 'U10',
                         'd', 'd',
                         'd')}
raw_data = np.loadtxt(data_file, skiprows=1, delimiter=',', dtype=data_type)


# %%

# Collect duration and image TTC for each case in sequence
num_images = np.max(raw_data['image'])
case_labels = {}
case_image_durations = {}
case_durations = {}
cameraTTC = {}
lidarTTC = {}

# TTC

# go through each data point
for row in raw_data:
    case = row['case']
    img = row['image'] - 1
    if case not in case_labels:
        case_labels[case] = (row['detector'], row['descriptor'])
        case_image_durations[case] = np.nan * np.ones((num_images, ))
        cameraTTC[case] = np.nan * np.ones((num_images, ))
        lidarTTC[case] = np.nan * np.ones((num_images, ))
    # add data
    case_image_durations[case][img] = row['processing_time']
    cameraTTC[case][img] = row['camera_ttc']
    lidarTTC[case][img] = row['lidar_ttc']

# get mean timing
for case in case_image_durations:
    case_durations[case] = np.mean(case_image_durations[case])

# %%
# Lidar TTC plot

# plot labels
plot_title = "Image vs. Lidar TTC"
plot_x_label = "Image number"
plot_y_label = "Lidar TTC (s)"

# plot data
image_numbers = np.arange(1, num_images + 1)

# line plot
fig_lidar_ttc, ax = plt.subplots()

# lidar
for case in lidarTTC:
    pline = ax.plot(image_numbers, lidarTTC[case], marker='o', label='Lidar', color='black', linewidth=2)[0]
    # Every lidar case is exactly the same, so no need to show all of them
    break

ax.set_xticks(image_numbers)
ax.set_ylabel(plot_y_label)
ax.set_xlabel(plot_x_label)
ax.set_title(plot_title)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %%
# Camera TTC plot

# plot labels
plot_title = "Image vs. TTC"
plot_x_label = "Image number"
plot_y_label = "TTC (s)"

# plot data
image_numbers = np.arange(1, num_images+1)

# line plot
fig_ttc, ax = plt.subplots()

# lidar
for case in lidarTTC:
    pline = ax.plot(image_numbers, lidarTTC[case], label='Lidar', color='black', linewidth=2)[0]
    # Every lidar case is exactly the same, so no need to show all of them
    break

# camera
for case in cameraTTC:
    if case_labels[case][0] in test_cases_to_plot:
        detector = case_labels[case][0]
        if case_labels[case][1] in test_cases_to_plot[detector]:
            descriptor = case_labels[case][1]
            label = "{}-{}".format(detector, descriptor)
            ax.plot(image_numbers, cameraTTC[case], label=label)

ax.set_xticks(image_numbers)
ax.set_ylabel(plot_y_label)
ax.set_xlabel(plot_x_label)
ax.set_title(plot_title)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %%
# timing plot - descriptor vs detector histogram
# plot labels
plot_title = "Total Duration (ms)"
plot_x_label = "Detector"
plot_y_label = "Descriptor"

# detector and descriptor labels
plot_detector_labels = []
plot_descriptor_labels = []
plot_detector_index = {}
plot_descriptor_index = {}
for case in case_labels:
    detector, descriptor = case_labels[case]
    if detector not in plot_detector_index:
        plot_detector_index[detector] = len(plot_detector_labels)
        plot_detector_labels.append(detector)
    if descriptor not in plot_descriptor_index:
        plot_descriptor_index[descriptor] = len(plot_descriptor_labels)
        plot_descriptor_labels.append(descriptor)

# 2d mesh plot data
x = np.arange(-0.5, len(plot_detector_labels) + 0.5, 1.0)
y = np.arange(-0.5, len(plot_descriptor_labels) + 0.5, 1.0)

xx, yy = np.meshgrid(x, y)
zz = np.zeros(xx.shape)
zz = zz[:-1, :-1]

minmax = None
for case in case_labels:
    detector, descriptor = case_labels[case]
    xind = plot_detector_index[detector]
    yind = plot_descriptor_index[descriptor]
    zz[yind, xind] = case_durations[case]
    if minmax is None:
        minmax = [case_durations[case], case_durations[case]]
    else:
        minmax = [np.fmin(minmax[0], case_durations[case]),
                  np.fmax(minmax[1], case_durations[case])]
zz[zz < minmax[0]] = minmax[1] + 0.5

# plot
fig_duration_mesh, ax = plt.subplots()
im = ax.pcolor(xx, yy, zz, vmin=minmax[0], vmax=step_size*1000, cmap='rainbow')
fig_duration_mesh.colorbar(im, ax=ax)

ax.set_xticks(x[:-1] + 0.5)
ax.set_xticklabels(plot_detector_labels)

ax.set_yticks(y[:-1] + 0.5)
ax.set_yticklabels(plot_descriptor_labels)

for case in case_labels:
    detector, descriptor = case_labels[case]
    xind = plot_detector_index[detector]
    yind = plot_descriptor_index[descriptor]
    ax.annotate("{:.0f}".format(case_durations[case]), (xind - 0.2, yind))

ax.set_ylabel(plot_y_label)
ax.set_xlabel(plot_x_label)
ax.set_title(plot_title)
# %%
# Duration table

# table_data = []
# rows = []
# columns = ('Detector', 'Descriptor', 'Duration (ms)')

# durations = []
# for case in case_labels:
#     detector, descriptor = case_labels[case]

#     durations.append(case_durations[case])
#     duration_string = "{:.1f}".format(case_durations[case])

#     table_data.append([detector, descriptor, duration_string])
#     rows.append("Case {}".format(case))

# columns = np.array(columns)
# rows = np.array(rows)
# table_data = np.array(table_data)

# sort_indices = np.argsort(durations)
# rows = rows[sort_indices]
# table_data = table_data[sort_indices, :]

# fig_table_sorted, ax = plt.subplots()
# ax.axis('tight')
# ax.axis('off')
# ax.table(cellText=table_data,
#          rowLabels=rows,
#          colLabels=columns, loc='center')

# %%
# save plots

fprefix, _ = os.path.splitext(data_file)

fig_lidar_ttc.set_size_inches(plot_size['width'], plot_size['height'])
fig_ttc.set_size_inches(plot_size['width'], plot_size['height'])
fig_duration_mesh.set_size_inches(plot_size['width'], plot_size['height'])

fig_lidar_ttc.tight_layout()
fig_ttc.tight_layout()
fig_duration_mesh.tight_layout()

print("Saving plots...")

fig_lidar_ttc.savefig(fprefix + "_lidar_ttc.png",
                       bbox_inches='tight', dpi=plot_size['dpi'])
fig_ttc.savefig(fprefix + "_ttc.png",
                       bbox_inches='tight', dpi=plot_size['dpi'])
fig_duration_mesh.savefig(fprefix + "_duration_mesh.png",
                          bbox_inches='tight', dpi=plot_size['dpi'])

print("Done")

# show plots
if display_plots:
    plt.show()
