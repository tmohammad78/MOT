import random

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_subject_name(sub_index):
    number_part = sub_index[3:]
    modified_number = number_part[:-1] if len(number_part) > 1 else '0'
    return f'sub{modified_number}'

def map_points_on_frame(frame):
    return round(frame * (7798 / 110))

def open_video(video):
    trial_video = cv2.VideoCapture(os.path.join('./videos', video.video_type, video.target_name + '.avi'))
    
    if not trial_video.isOpened():
        raise ValueError(f"Unable to open video file: {os.path.join('./videos', video.video_type, video.target_name + '.avi')}")
    return trial_video

def generate_x_points(x):
    # scale_x = 660 / 1920
    # return int(x * scale_x)
    return int(x) - 630

def generate_y_points(y):
    return int(y) - 210

def make_ready_data_to_draw(video):
    return {
        'first': video.first_video,
        'second': video.second_video,
        'firstTargets': video.firstTargets,
        'secondTargets': video.secondTargets,
        'firstStart': video.first_start_frame,
        'secondStart': video.second_start_frame
    }
