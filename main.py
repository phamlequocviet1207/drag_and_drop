# https://techtutorialsx.com/2021/04/10/python-hand-landmark-estimation/


import mediapipe as mp
import cv2
import math

#Define the dimension of Window Screen
widthScreen = 640
heightScreen = 480

#Define the location of the box, the color, and the dimension
xCenBlock = widthScreen//2
yCenBlock = heightScreen//2

wBlock = 100
hBlock = 100

upleft = (xCenBlock - wBlock // 2,yCenBlock - hBlock //2)
botright = (xCenBlock + wBlock // 2,yCenBlock + hBlock //2)

colorB = (255,255,0)

#import modules needed
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

#Capture video
cap = cv2.VideoCapture(0)

#Set the dimension of the Window Screen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, widthScreen)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, heightScreen)

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=2) as hands:

    while True:
        ret, frame = cap.read()

        results = hands.process(frame)

        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
                
            # Index finger tip = 8; thumb tip= 4
            if handsModule.HandLandmark(4) and handsModule.HandLandmark(8) and handsModule.HandLandmark(12):
                point4 = handsModule.HandLandmark(4)
                point8 = handsModule.HandLandmark(8)
                point12 = handsModule.HandLandmark(12)

                normalizedLandmark4 = handLandmarks.landmark[point4]
                x4, y4 = pixelCoordinatesLandmark4 = drawingModule._normalized_to_pixel_coordinates(
                    normalizedLandmark4.x,
                    normalizedLandmark4.y,
                    widthScreen,
                    heightScreen)

                normalizedLandmark8 = handLandmarks.landmark[point8]
                x8, y8 = pixelCoordinatesLandmark8 = drawingModule._normalized_to_pixel_coordinates(
                    normalizedLandmark8.x,
                    normalizedLandmark8.y,
                    widthScreen,
                    heightScreen)

                normalizedLandmark12 = handLandmarks.landmark[point12]
                x12, y12 = pixelCoordinatesLandmark12 = drawingModule._normalized_to_pixel_coordinates(
                    normalizedLandmark12.x,
                    normalizedLandmark12.y,
                    widthScreen,
                    heightScreen)

                # Emphasis 2 tips of 2 fingers: index and thumb
                cv2.circle(frame, pixelCoordinatesLandmark4, 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, pixelCoordinatesLandmark8, 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, pixelCoordinatesLandmark12, 10, (255, 0, 255), cv2.FILLED)

                # Find the distance between 2 fingers
                distance4_8 = math.sqrt((x4-x8)**2 + (y4-y8)**2)
                distance8_12 = math.sqrt((x12 - x8) ** 2 + (y12 - y8) ** 2)
                print(distance8_12)

                # print(x4,y4)
                # print(x8,y8)

                if (upleft[0] <= x8 <= botright[0] and upleft[1] <= y8 <= botright[1] and upleft[0] <= x12 <= botright[0] and upleft[1] <= y12 <= botright[1]):
                    colorB = (0,255,255)

                    # Move the box with the index finger tip
                    if (distance8_12 < 30):
                        xCenBlock, yCenBlock = x8, y8
                        upleft = (xCenBlock - wBlock // 2, yCenBlock - hBlock // 2)
                        botright = (xCenBlock + wBlock // 2, yCenBlock + hBlock // 2)
                else:
                    colorB = (255,255,0)

        # Draw a rectangle
        cv2.rectangle(frame, upleft, botright, colorB, cv2.FILLED)

        cv2.imshow("Screen", frame)
        cv2.moveWindow("Screen", 400,200)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()