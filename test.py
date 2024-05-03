import face_recognition
import cv2
import numpy as np 
import argparse
import beepy
import pywhatkit as kit
import time

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=True)
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

def start_webcam():
    cap = cv2.VideoCapture(0)

    return cap

#Load yolo
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

# Load a second sample picture and learn how to recognize it.
bradley_image = face_recognition.load_image_file("bradley.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

# Load a third sample picture and learn how to recognize it.
jayesh_image = face_recognition.load_image_file("Jayesh.jpg")
jayesh_face_encoding = face_recognition.face_encodings(jayesh_image)[0]

# Load a fourth sample picture and learn how to recognize it.
indresh_image = face_recognition.load_image_file("indresh.jpg")
indresh_face_encoding = face_recognition.face_encodings(indresh_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    bradley_face_encoding,
    jayesh_face_encoding,
    indresh_face_encoding
]
known_face_names = [
    "Bradley",
    "jayesh",
    "Indresh"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []




def display_blob(blob):
    '''
        Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids
            
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    print(indexes)
    # indexes=np.arange(1)
    if(indexes==0):
        print("weapon detected in frame")
    # if(indexes==0):
    #     for i in range(0,7):
    #         beepy.beep(sound=1)
    
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i-1]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    img=cv2.resize(img, (800,600))
    cv2.imshow("Image", img)



def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        _, img = cap.read()
        height, width, channels = img.shape
        width = 512
        height = 512
        process_this_frame = True
        if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
        
            for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]
            
            

            # Or instead, use the known face with the smallest distance to the new face
                Unkown,image=cap.read()
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # kit.sendwhatmsg_instantly("+918277469493","someone has entered your house")
                # if(name=="Unknown"):
                #     for i in range(0,7):
                #        beepy.beep(sound=1)
                # if name=="Unknown":
                #     cv2.imwrite("Stranger.jpg",images)
                #     kit.sendwhatmsg_instantly("+918277469493","Stranger has entered your house")
                #     kit.sendwhats_image("+918277469493", "Stranger.jpg")

                face_names.append(name)

        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

        # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()



if __name__ == '__main__':
    webcam = args.webcam
    if webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        webcam_detect()

    cv2.destroyAllWindows()