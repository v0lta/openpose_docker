import cv2
import sys
import imageio
import ipdb
import numpy as np
import matplotlib.pyplot as plt
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


plot = True
write_movie = True
if write_movie:
    writer = imageio.get_writer('./orientation.mp4', fps=25)

for frame_no, frame in enumerate(vid):

    depth_frame = vid_depth.get_next_data()[..., 0]
    depth_frame_norm = depth_frame/np.max(depth_frame)
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
    print(frame_no)
    if frame_no > 25:

        plt.imshow(frame)
        plt.savefig('frame')

        heatmaps = datum.poseHeatMaps.copy()
        heatmaps = (heatmaps).astype(dtype='uint8')

        ls_heat = heatmaps[2, :, :]
        rs_heat = heatmaps[5, :, :]
        rh_heat = heatmaps[9, :, :]
        lh_heat = heatmaps[12, :, :]

        depth = cv2.resize(depth_frame, (heatmaps.shape[2],
                                         heatmaps.shape[1]))

        if plot:
            outputImageF = (datum.inputNetData[0].copy())[0, :, :, :] + 0.5
            outputImageF = cv2.merge([outputImageF[0, :, :],
                                      outputImageF[1, :, :],
                                      outputImageF[2, :, :]])
            outputImageF = (outputImageF*255.).astype(dtype='uint8')

            num_maps = heatmaps.shape[0]
            for counter in range(num_maps):
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
                ipdb.set_trace()
                counter += 1
                counter = counter % num_maps

        ipdb.set_trace()
