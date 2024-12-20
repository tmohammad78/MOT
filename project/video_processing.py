import os
import cv2
import json
import numpy as np

def parse_video_name(video_name):
    video_name = video_name.replace('.avi', '')
    first_part, second_part = video_name.split(',')
    video_type, num_targets = first_part.rsplit('_', 1)
    videos_info = second_part.split('_')
    
    parsed_data = {
        'video_name': video_name,
        'target_name': second_part,
        'video_type': video_type,
        'num_targets': int(num_targets),
        'first_video': int(videos_info[0]),
        'second_video': int(videos_info[1]),
        'first_start_frame': int(videos_info[2]),
        'second_start_frame': int(videos_info[3]),
        'first_rotation': float(videos_info[4]),
        'second_rotation': float(videos_info[5])
    }
    
    return parsed_data

def load_shape_fish_points(video_name, number_frame):
    json_filename = f'img{number_frame + 1:03d}.json'
    json_path = os.path.join('./jsons', str(video_name), json_filename)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    return data.get('shapes', [])

def get_targets_from_json(shapes, targets):
    if not isinstance(targets, (list, tuple)):
        targets = [targets]
    
    target_digits = set()
    for target in targets:
        if isinstance(target, int):
            target_digits.update(str(target))
        else:
            raise ValueError("Targets should be integers or lists/tuples of integers")

    filtered_shapes = [shape for shape in shapes if shape['label'] in target_digits]
    
    return filtered_shapes


def get_total_shape_first(shapeInfo,firstFrame):
    firstTargets = shapeInfo['firstTargets']
    shapes_video1 = loadShapeFishPoints(shapeInfo['first'], firstFrame)
    return getTargetsFromJson(shapes_video1,firstTargets)
    
def get_total_shape_second(shapeInfo,secondFrame):
    secondTargets = shapeInfo['secondTargets']
    shapes_video2 = loadShapeFishPoints(shapeInfo['second'], secondFrame)
    merged_shapes = getTargetsFromJson(shapes_video2,secondTargets)
    return merged_shapes
    

def get_video_info(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height 

def create_output_video(video, name):
    output_gaze_videos = 'gazeOutputVideo'
    os.makedirs(output_gaze_videos, exist_ok=True)
    fps, width, height = get_video_info(video)
    
    output_gaze_videos_path = os.path.join(output_gaze_videos, name + '.avi')
    output_video = cv2.VideoWriter(output_gaze_videos_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    return output_video

def rotate_frame(frame, angle, center):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return rotated_frame

def adjust_shape_points(shape, x_offset=90, y_offset=10):
    adjusted_points = []
    for point in shape['points']:
        adjusted_point = [point[0] - x_offset, point[1] - y_offset]
        adjusted_points.append(adjusted_point)
    shape['points'] = adjusted_points
    return shape

def draw_shapes(frame, shapes, type_v, color=(0, 0, 255)):
    data = {
        1: [90, 5],
        7: [25, 25],
        10: [25, 15]
    }
    overlay = frame.copy()
    opacity = 0.1
    for shape in shapes:
        shape = adjust_shape_points(shape, data[type_v][0], data[type_v][1])
        points = shape['points']

        if not points:
            continue
        
        np_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        if np_points.size == 0:
            continue

        cv2.polylines(overlay, [np_points], isClosed=True, color=color, thickness=1)
        cv2.fillPoly(overlay, [np_points], color=color)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
