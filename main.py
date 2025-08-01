import cv2 as cv
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
"""
1. We need to first receive input from the webcam 
2. OpenCV seems to display the footage reversed so now we need to flip it
3. Detect footage using mp
4. Print mp's results and create an algorithm to detect whether its a left or right hand
"""

Base_options = mp.tasks.BaseOptions
hand_landmarker = mp.tasks.vision.HandLandmarker
hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
hand_landmarker_result = mp.tasks.vision.HandLandmarkerResult
vision_running_mode = mp.tasks.vision.RunningMode
model_path = r"C:\Users\rowdh\OneDrive\Documents\Tomfoolery\Personal Project\face-detector\hand_landmarker.task"

# Captures the video from the web cam using opencv
cam = cv.VideoCapture(0)
latest_results = [""]
def print_result(result: hand_landmarker_result, output_image, timestamp_ms: int):
    latest_results.append(result.handedness)
    return(result)

options = hand_landmarker_options(
    base_options = Base_options(model_asset_path = model_path),
    num_hands = 2,
    running_mode = vision_running_mode.LIVE_STREAM,
    result_callback = print_result
)
detector = hand_landmarker.create_from_options(options)
while cam:
    # bool is whether or not a frame was captured. 
    # frame is the numpy array
    bool, frame = cam.read()

    # inverses the frames to mirror what you see irl
    flipped_frame = cv.flip(frame, 1)
    timestamp_msec = int(cam.get(0))    
    rgb = cv.cvtColor(flipped_frame,4)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = detector.detect_async(mp_image, timestamp_msec)
    cv.imshow("hand detector 9000",flipped_frame)
   # print(latest_results)
    if cv.waitKey(1) == ord("q"):
        break

    #[
    # [Category(index=1, score=0.9256862998008728, display_name='Left', category_name='Left')], 
    # [Category(index=0, score=0.7756921052932739, display_name='Right', category_name='Right')]
    #]

    # 3 nested lists:
    # [-1]: [[Category(index=1, score=0.5256191492080688, display_name='Left', category_name='Left')]]
    # [-1][-1]:[Category(index=1, score=0.5256191492080688, display_name='Left', category_name='Left')]
    # [-1][-1][-1]: Left
    if latest_results[-1]:
        last_result = latest_results[-1]
        if len(last_result) == 2:
            print("Two Hands")
        elif len(last_result) == 1:
            hand_1 = last_result[-1][-1].display_name
            if hand_1 == "Left":
                print("Right")
            elif hand_1 == "Right":
                print("Left")

cam.release()
cv.destroyAllWindows()