import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import multiprocessing
from multiprocessing import Pool
import glob
import fnmatch


#Videos:
PATH = '/Users/josegarcia/Documents/GitHub/Sign-Language-Capstone/Data'
list_of_videos = []
for file in os.listdir(PATH):
    if fnmatch.fnmatch(file, '*.mp4'):
        list_of_videos.append(file)


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB- required in Mediapipe
    image.flags.writeable = False                  # Image is no longer writeable, saves memory
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

#cap = cv2.VideoCapture(0)
#mp4 = '/Users/josegarcia/Documents/GitHub/Sign-Language-Capstone/Data/_-adcxjm1R4_0-8-rgb_front.mp4'
#cap = cv2.VideoCapture(mp4)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    vectors = np.concatenate([pose, face, lh, rh]) 
    return vectors

# Set mediapipe model 
def process_video(video):
    try: 
        os.makedirs(os.path.join(PATH+'/vectors/', video))
    except:
        pass

    cap = cv2.VideoCapture(PATH+'/{}'.format(video))
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            print([frame_num])
            # Read feed
            ret, frame = cap.read()
            if ret == True:

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)

                #Export Keypoints
                keypoints = extract_keypoints(results)
                desired_dir = PATH+'/vectors/{}'.format(video)
                full_path = os.path.join(desired_dir,str(frame_num))
                np.save(full_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()


def wrapper(video):
    process_video(video)
    


print('starting the script')

print('Processing ', len(list_of_videos), 'number of videos')
if __name__ == '__main__':
    with multiprocessing.get_context("spawn").Pool() as pool:
        p = Pool(8)
        p.map(wrapper, list_of_videos)



