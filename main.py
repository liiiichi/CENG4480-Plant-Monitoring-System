import cv2
import imutils.contours
import numpy as np
from datetime import datetime
import os
from time import sleep
import json
import RPi.GPIO as GPIO
import threading
from fastiecm import fastiecm

# Constants
PUMP_PIN = 17
SOIL_SENSOR_PIN = 18
NDVI_FOLDER = '/home/chi/proj/ndvi_images'
COLORED_FOLDER = '/home/chi/proj/color_images'
UNHEALTHY_THRESHOLD = 150

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUMP_PIN, GPIO.OUT)
GPIO.setup(SOIL_SENSOR_PIN, GPIO.IN)

def log_plant_data(plant_height, avg_ndvi, is_dry):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "plant_height_mm": plant_height,
        "avg_ndvi": avg_ndvi,
        "soil_moisture_is_dry": is_dry
    }
    with open('plant_data_log.json', 'a') as file:
        json.dump(log_entry, file)
        file.write('\n')

# Function for controlling the pump based on soil moisture
def control_pump():
    while True:
        is_dry = GPIO.input(SOIL_SENSOR_PIN)
        GPIO.output(PUMP_PIN, GPIO.LOW if is_dry else GPIO.HIGH)
        print(f"Soil Moisture: {'Dry' if is_dry else 'Wet'}")
        sleep(5)

# NDVI calculation and image capture functions
def capture_image():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    return None

def display(image, image_name, avg_ndvi):
	image = np.array(image, dtype=float)/float(255)
	shape = image.shape
	height = int(shape[0] / 2)
	width = int(shape[1] / 2)
	image1 = cv2.resize(image, (width, height))
	cv2.putText(image1, f"avg_ndvi: {avg_ndvi:.2f}", (50,25),
			  cv2.FONT_HERSHEY_SIMPLEX,0.5,(55,255,155),1)
	cv2.namedWindow(image_name)
	cv2.imshow(image_name, image1)



def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

def save_image(image, folder, prefix, timestamp):
    filename = f"{prefix}_{timestamp}.png"
    cv2.imwrite(os.path.join(folder, filename), image)

# Analysis and logging functions
def calculate_average_ndvi(image):
    return np.mean(image)

def image_process():
	# Define height range in millimeters
	MIN_HEIGHT_MM = 20
	MAX_HEIGHT_MM = 150

	# Initialize the camera
	cap = cv2.VideoCapture(0)

	if not cap.isOpened():
		print("Failed to open camera.")
		exit()

	last_valid_height = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			print("Failed to capture frame")
			break

		# Convert to grayscale and blur
		greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		greyscale = cv2.GaussianBlur(greyscale, (7, 7), 0)

		# Detect edges
		canny_output = cv2.Canny(greyscale, 50, 100)
		canny_output = cv2.dilate(canny_output, None, iterations=1)
		canny_output = cv2.erode(canny_output, None, iterations=1)

		# Find contours
		contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if len(contours) < 2:
			print("Couldn't detect two or more objects")
			continue

		(contours, _) = imutils.contours.sort_contours(contours)
		contours_poly = [None] * len(contours)
		boundRect = [None] * len(contours)
		for i, c in enumerate(contours):
			contours_poly[i] = cv2.approxPolyDP(c, 3, True)
			boundRect[i] = cv2.boundingRect(contours_poly[i])

		output_image = frame.copy()
		mmPerPixel = 15 / boundRect[0][2]  # Assuming 10mm is the known width of the first object
		highestRect = 1000
		lowestRect = 0

		for i in range(1, len(contours)):
			if boundRect[i][2] < 50 or boundRect[i][3] < 50:
				continue

			if highestRect > boundRect[i][1]:
				highestRect = boundRect[i][1]
			if lowestRect < (boundRect[i][1] + boundRect[i][3]):
				lowestRect = (boundRect[i][1] + boundRect[i][3])

			cv2.rectangle(output_image, (int(boundRect[i][0]), int(boundRect[i][1])),
						  (int(boundRect[i][0] + boundRect[i][2]),
						   int(boundRect[i][1] + boundRect[i][3])), (255, 0, 0), 2)
		#ndvi
		contrasted = contrast_stretch(frame)
		ndvi = calc_ndvi(contrasted)
		ndvi_contrasted = contrast_stretch(ndvi)
		color_mapped_prep = ndvi_contrasted.astype(np.uint8)
		color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)



		# Analyze NDVI and log
		avg_ndvi = calculate_average_ndvi(ndvi_contrasted)
		#print(f"avg_ndvi: {avg_ndvi}")

		plantHeight = (lowestRect - highestRect) * mmPerPixel

		#get time
		time_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

		# Check if the plant height is within the valid range
		if (MIN_HEIGHT_MM <= plantHeight <= MAX_HEIGHT_MM) & (0 <= avg_ndvi <= 255) :
			last_valid_height = plantHeight
			print(f"Plant height is {plantHeight:.0f}mm  Average NDVI: {avg_ndvi:.2f} Time: {time_text}")
		else :
			plantHeight = last_valid_height
		#	print(f"Plant height is {plantHeight:.0f}mm  avg_ndvi: {avg_ndvi}")
		#   print("Using last valid height.")


		#display ndvi image
		display(color_mapped_image, "Color mapped" , avg_ndvi)
		cv2.putText(color_mapped_image, f"Average NDVI: {avg_ndvi:.2f}  Time: {time_text}", (64,30),
			  cv2.FONT_HERSHEY_SIMPLEX,0.5,(55,255,155),1)
		# Display the image
		resized_image = cv2.resize(output_image, (640, 360))
		cv2.putText(resized_image, f"Plant height:{plantHeight:.0f}mm  Time: {time_text}", (60,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,0,0),1)
		cv2.imshow("Image", resized_image)
		cv2.imwrite('ndvi.png', color_mapped_image)
		cv2.imwrite('plant_height.png', resized_image)
		is_dry = GPIO.input(SOIL_SENSOR_PIN)
		log_plant_data(plantHeight, avg_ndvi, is_dry)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		sleep(30)

	cap.release()
	cv2.destroyAllWindows()

# Multithreading
if __name__ == "__main__":
    pump_thread = threading.Thread(target=control_pump)
    image_process_thread = threading.Thread(target=image_process)

    pump_thread.start()
    image_process_thread.start()

    pump_thread.join()
    image_process_thread.join()