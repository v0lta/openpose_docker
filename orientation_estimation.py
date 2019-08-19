import cv2
import sys
import time
import imageio
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('/usr/local/python')
import openpose.pyopenpose as op

params = dict()
params['model_folder'] = '/home/openpose/models/'
params['model_pose'] = "BODY_25"
params["heatmaps_add_parts"] = True
params["heatmaps_add_bkg"] = True
params["heatmaps_add_PAFs"] = True
params["heatmaps_scale"] = 2
# params[""]

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

filename = '/home/video_input/Nils1/Nils1_with_frame_numbers.mp4'
# open the depth video as well.
filename_depth = '/home/video_input/Nils1/Nils_Depth1_with_frame_numbers.mp4'


vid = imageio.get_reader(filename, 'ffmpeg')
vid_depth = imageio.get_reader(filename_depth, 'ffmpeg')


plot = False
write_movie = True
if write_movie:
    writer = imageio.get_writer('./orientation.mp4', fps=25)

for frame_no, frame in enumerate(vid):

    start = time.time()

    depth_frame = vid_depth.get_next_data()[..., 0]
    depth_frame_norm = depth_frame/np.max(depth_frame)

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)

    stop = time.time()
    print('frame', frame_no, 'processed. In:', stop-start)

    if frame_no > 0:

        # plt.imshow(frame)
        # plt.savefig('frame')

        heatmaps = datum.poseHeatMaps.copy()
        heatmaps = (heatmaps).astype(dtype='uint8')

        joint_total = 25
        ls_heat = heatmaps[2, :, :]
        rs_heat = heatmaps[5, :, :]
        rh_heat = heatmaps[9, :, :]
        lh_heat = heatmaps[12, :, :]

        depth_frame_norm = cv2.resize(depth_frame_norm, (heatmaps.shape[2],
                                                         heatmaps.shape[1]))

        if plot:
            outputImageF = (datum.inputNetData[0].copy())[0, :, :, :] + 0.5
            outputImageF = cv2.merge([outputImageF[0, :, :],
                                      outputImageF[1, :, :],
                                      outputImageF[2, :, :]])
            outputImageF = (outputImageF*255.).astype(dtype='uint8')

            num_maps = heatmaps.shape[0]
            for counter in range(joint_total):
                # map position - joint
                # 2,  Left-Shoulder,
                # 5,  Right-Shoulder,
                # 9,  Right-Hip,
                # 12, Left-Hip

                heatmap = heatmaps[counter, :, :].copy()
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                combined = cv2.addWeighted(outputImageF, 0.5, heatmap_color, 0.5, 0)
                print('counter', counter)
                plt.imshow(combined)
                plt.savefig('combined.png')
                # ipdb.set_trace()
                counter += 1

        # compute joint positions.]
        joints_xyz = []
        zero_canvas = heatmaps[0, :, :]*0.0
        # plt.imshow(to_plot)
        # plt.show()
        for joint_no in range(joint_total):
            # print(joint_no)
            # zero_canvas = zero_canvas + heatmap[:, :, joint_no]
            # plt.imshow(zero_canvas)
            # plt.show()
            nx, ny = heatmaps.shape[1:]
            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)
            x_grid, y_grid = np.meshgrid(y, x)
            y_grid = np.flipud(y_grid)

            def mean_coordinate(coords, heat, thre1=0.1):
                heat = np.where(heat > thre1, heat, 0)
                coords = coords * heat
                return np.sum(coords.flatten())/np.sum(heat.flatten())

            mean_x = mean_coordinate(x_grid, heatmaps[joint_no, :, :])
            mean_y = mean_coordinate(y_grid, heatmaps[joint_no, :, :])
            mean_z = mean_coordinate(depth_frame_norm, heatmaps[joint_no, :, :])
            joints_xyz.append((mean_x, mean_y, mean_z))
            joints_xyz_array = np.array(joints_xyz)
            # plt.plot(joints_xyz_array[:, 0], joints_xyz_array[:, 1], '.')
            # plt.show()

        joints_xyz = np.array(joints_xyz)

        # plt.plot(joints_xyz[1:14, 0], joints_xyz[1:14, 1], '.')
        # plt.show()

        # heat depth-plot.
        heat_sum = ls_heat + rs_heat + rh_heat + lh_heat
        heat_sum_norm = heat_sum / np.max(heat_sum)
        heat_sum_norm = (heat_sum_norm*255).astype(dtype='uint8')
        depth_frame_norm = np.stack([depth_frame_norm,
                                     depth_frame_norm,
                                     depth_frame_norm], -1)
        depth_frame_norm = (depth_frame_norm*255).astype(dtype='uint8')
        heatmap_color = cv2.applyColorMap(heat_sum_norm, cv2.COLORMAP_JET)
        combined_depth = cv2.addWeighted(depth_frame_norm, 0.5, heatmap_color, 0.5, 0)

        # ipdb.set_trace()
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.imshow(datum.cvOutputData)
        plt.axis('off')

        ax = fig.add_subplot(132)
        ax.imshow(combined_depth)
        plt.axis('off')

        ax = fig.add_subplot(133, projection='3d')
        left_shoulder = joints_xyz[5, :]
        right_shoulder = joints_xyz[2, :]
        left_hip = joints_xyz[11, :]
        right_hip = joints_xyz[8, :]

        shoulder_hip_rectangle = np.array([left_hip, right_hip,
                                           left_shoulder, right_shoulder])
        ax.scatter(shoulder_hip_rectangle[:, 0],
                   shoulder_hip_rectangle[:, 2],
                   shoulder_hip_rectangle[:, 1], c='b')

        # compute all 4 normal vectors for the plane.
        lh_rh = left_hip - right_hip
        lh_ls = left_hip - left_shoulder
        lh_rh_ls_n = np.cross(lh_rh, lh_ls)

        rh_rs = right_hip - right_shoulder
        rh_lh = right_hip - left_hip
        rh_lh_rs_n = np.cross(rh_rs, rh_lh)

        ls_lh = left_shoulder - left_hip
        ls_rs = left_shoulder - right_shoulder
        ls_rh_ls_n = np.cross(ls_lh, ls_rs)

        rs_ls = right_shoulder - left_shoulder
        rs_rh = right_shoulder - right_hip
        rs_ls_rh_n = np.cross(rs_ls, rs_rh)

        center = np.mean(shoulder_hip_rectangle, axis=0)
        normals = np.array([lh_rh_ls_n, rh_lh_rs_n, ls_rh_ls_n, rs_ls_rh_n])
        normal = np.mean(normals, axis=0)
        tau = 3
        norm_vec = np.array([center, center + tau*normal])
        ax.plot(norm_vec[:, 0],
                norm_vec[:, 2],
                norm_vec[:, 1], c='r')
        ax.scatter(norm_vec[-1, 0],
                   norm_vec[-1, 2],
                   norm_vec[-1, 1], c='r', s=60)

        if write_movie:
            plt.axis('off')
            fig.canvas.draw()
            # ipdb.set_trace()
            to_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            to_plot = to_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(to_plot)
            plt.close()
        else:
            plt.show()
