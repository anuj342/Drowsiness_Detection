from scipy.spatial import distance
import imutils
import cv2
import numpy as np

# Using OpenCV's pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25
frame_check = 20
cap=cv2.VideoCapture(0)
flag=0
while True:
	ret, frame=cap.read()
	if not ret:
		break
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	for (x, y, w, h) in faces:
		# Detect eyes within face region
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		
		eyes = eye_cascade.detectMultiScale(roi_gray)
		
		if len(eyes) >= 2:
			# Sort eyes by x coordinate (left to right)
			eyes = sorted(eyes, key=lambda e: e[0])
			
			# Get left and right eyes
			leftEye_x, leftEye_y, leftEye_w, leftEye_h = eyes[0]
			rightEye_x, rightEye_y, rightEye_w, rightEye_h = eyes[1]
			
			# Create approximate 6-point landmarks for each eye (3 points top, 3 points bottom)
			left_eye = np.array([
				[leftEye_x, leftEye_y + leftEye_h//3],                    # left point
				[leftEye_x + leftEye_w//2, leftEye_y],                   # top-left
				[leftEye_x + leftEye_w, leftEye_y],                      # top-right
				[leftEye_x + leftEye_w, leftEye_y + leftEye_h//2],       # right point
				[leftEye_x + leftEye_w//2, leftEye_y + leftEye_h],       # bottom-right
				[leftEye_x, leftEye_y + leftEye_h//2]                    # bottom-left
			], dtype=np.int32)
			
			right_eye = np.array([
				[rightEye_x, rightEye_y + rightEye_h//3],                 # left point
				[rightEye_x + rightEye_w//2, rightEye_y],                # top-left
				[rightEye_x + rightEye_w, rightEye_y],                   # top-right
				[rightEye_x + rightEye_w, rightEye_y + rightEye_h//2],   # right point
				[rightEye_x + rightEye_w//2, rightEye_y + rightEye_h],   # bottom-right
				[rightEye_x, rightEye_y + rightEye_h//2]                 # bottom-left
			], dtype=np.int32)
			
			# Adjust coordinates to frame space
			left_eye[:, 0] += x
			left_eye[:, 1] += y
			right_eye[:, 0] += x
			right_eye[:, 1] += y
			
			# Calculate eye aspect ratio
			leftEAR = eye_aspect_ratio(left_eye)
			rightEAR = eye_aspect_ratio(right_eye)
			ear = (leftEAR + rightEAR) / 2.0
			
			# Draw eye contours
			leftEyeHull = cv2.convexHull(left_eye)
			rightEyeHull = cv2.convexHull(right_eye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			
			# Check for drowsiness
			if ear < thresh:
				flag += 1
				print (flag)
				if flag >= frame_check:
					cv2.putText(frame, "****************ALERT!****************", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					cv2.putText(frame, "****************ALERT!****************", (10,325),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					#print ("Drowsy")
			else:
				flag = 0
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
cap.release() 
