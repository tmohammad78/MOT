import os
import cv2
import pandas as pd
import numpy as np
import json
from video_processing import *
from data_loading import *
from utils import *

subject_colors = {}

def checkPointValid(point, previous_point, frame_idx, axis):
    if math.isnan(point):
        if previous_point is not None and not math.isnan(previous_point):
            print(f"NaN detected for {axis} at frame {frame_idx}. Using previous {axis} value: {previous_point}")
            return previous_point
        else:
            print(f"NaN detected for {axis} at frame {frame_idx}, but no previous value to replace. Skipping.")
            return 0
    return point
    
def rotate_frameNew(frame, angle):
    center = (330, 330)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return rotated_frame


def get_subject_color(subject):
    if subject not in subject_colors:
        subject_colors[subject] = generate_random_color()
    return subject_colors[subject]

if __name__ == "__main__":
    subjects = load_mat_file()
    data = get_subject_trial_info()

    videoList = []
    trialNumber  = 0
    
    for subjectTrialNumber, trialInfo in data.items():
        previous_x = None
        previous_y = None
        
        for index, video in trialInfo.iterrows():
            try:
                # print(index,'this is index')
                open_video = openVideo(video)
                output_video_name = f"{subjectTrialNumber}_{video.video_name}"
                output_video = createOutputVideo(open_video,output_video_name)
                
                frame_idx = 0
        
                while True:
                    # print(trialNumber,'this is trialNumber *****************')
                    ret, frame = open_video.read()
                    if not ret:
                        break  
                    frame = rotate_frameNew(frame, video['fullRotation'])
                    for subIndex in subjects:
                        try:             
                            # print(subIndex,frame_idx,'subIndex')
                            x_points = subjects[subIndex][trialNumber][0]  # (110, 1), trial id (0 - 39) , (x ,y) , frame of trial ( 0 - 109)
                            y_points = subjects[subIndex][trialNumber][1]  # (110, 1)
            
                            # the index 0 is because of the structure, they have just one item 
                            x = checkPointValid(x_points[frame_idx][0], previous_x, frame_idx, 'x')
                            y = checkPointValid(y_points[frame_idx][0], previous_y, frame_idx, 'y')
        
                            if x is None or y is None:
                                continue  # Skip this frame if x or y is None
                                
                            previous_x = x
                            previous_y = y
        
        
                            new_x = generateXPoints(x)
                            new_y = generateYPoints(y)
                            
                            # print(subIndex,trialNumber,frame_idx,'x:',new_x,'y:',new_y)
                            color = get_subject_color(subIndex)
            
                            cv2.circle(frame, (new_x, new_y), 5, color, 3) 
            
                            cv2.putText(frame, generateSubjectName(subIndex), (new_x + 5, new_y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        except Exception as e:
                            print(f"An error occurred index {subIndex}: {e}")
                    #### Draw a shape for targets 
                    shapeInfo = {
                        'first': video.first_video,
                        'second': video.second_video,
                        'firstTargets': video.firstTargets,
                        'secondTargets': video.secondTargets,
                        'firstStart': video.first_start_frame,
                        'secondStart': video.second_start_frame
                    }
        
                    shapeFirst = get_total_shape_first(shapeInfo, shapeInfo['firstStart'] + frame_idx)
                    shapeSecond = get_total_shape_second(shapeInfo, shapeInfo['secondStart'] + frame_idx)
                    
                    blank_frame = np.zeros_like(frame)
                    blank_frame2 = np.zeros_like(frame)
                    draw_shapes(blank_frame, shapeFirst,video.first_video)
                    draw_shapes(blank_frame2, shapeSecond,video.second_video)
        
                    center = (330,330)
                    rotated_shapes_frame = rotate_frame(blank_frame, video.first_rotation, center)
                    
                    full_rotated_first = rotate_frameNew(rotated_shapes_frame, video['fullRotation'])
                    rotated_shapes_frame2 = rotate_frame(blank_frame2, video.second_rotation, center)
                    full_rotated_second = rotate_frameNew(rotated_shapes_frame2, video['fullRotation'])
        
                    mask = full_rotated_first > 0
                    frame[mask] = full_rotated_first[mask]
        
                    mask2 = full_rotated_second > 0
                    frame[mask2] = full_rotated_second[mask2]
                    

                    ## Write shapes on videos
                    resized_frame = cv2.resize(frame, (660, 660))
                    output_video.write(resized_frame)
                    
                    frame_idx += 1
                
                trialNumber += 1
                
            except Exception as e:
                print(f"An error occurred while processing video at index {index}: {e}")
        
            finally:
                if 'open_video' in locals() and open_video.isOpened():
                    open_video.release()
                if 'output_video' in locals():
                    output_video.release()
                cv2.destroyAllWindows()
        