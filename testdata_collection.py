# -*- coding: utf-8 -*-
# %%

# Import necessary libraries
from PIL import ImageFont, ImageDraw, Image
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
import keyboard
from my_functions import *
from language_selection_gui import select_language

# Function to display text (both English and Korean) on the OpenCV window using PIL
def display_text(image, text, position, font_size=30):
    font_path = "/Users/solo/Desktop/Sign-Language-Translator-main/NotoSerifKR-Black.otf"  # Make sure to provide a valid font path
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(0, 0, 255))
    image = np.array(img_pil)
    return image

languages = ['English', 'Korean']
selected_language = select_language(languages)

# Define the actions (signs) that will be recorded and stored in the dataset based on the selected language
if selected_language == 'Korean':
    actions = np.array(['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'])
else:
    actions = np.array(['A', 'B', 'C', 'D', 'E'])  # Example set of English alphabets

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
base_path = 'data'
PATH = os.path.join(base_path, selected_language)

if not os.path.exists(PATH):
    os.makedirs(PATH)

for action, sequence in product(actions, range(sequences)):
    os.makedirs(os.path.join(PATH, action, str(sequence)), exist_ok=True)


# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Loop through each action, sequence, and frame to record data
    for action, sequence, frame in product(actions, range(sequences), range(frames)):
        if frame == 0: 
            while True:
                if keyboard.is_pressed(' '):
                    break
                _, image = cap.read()
                if image is None:
                    continue 
                results = image_process(image, holistic)
                draw_landmarks(image, results)
                
                # Using PIL to display text for both English and Korean
                if selected_language == 'Korean':
                    image = display_text(image, f'Recording: "{action}". Sequence: {sequence}.', (20, 20))
                    image = display_text(image, 'Press "Space" when you are ready.', (20, 450), font_size=20)
                else:
                    image = display_text(image, f'Recording: "{action}". Sequence: {sequence}.', (20, 20))
                    image = display_text(image, 'Press "Space" when you are ready.', (20, 450), font_size=20)

                cv2.imshow('Camera', image)
                cv2.waitKey(1)

                if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                    break
        else:
            _, image = cap.read()
            if image is None:  
                continue 
            results = image_process(image, holistic)
            draw_landmarks(image, results)

            # Save the landmarks data for each frame
            keypoints = keypoint_extraction(results)
            frame_path = os.path.join(PATH, action, str(sequence), str(frame))
            np.save(frame_path, keypoints)

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()
