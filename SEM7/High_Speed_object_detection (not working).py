import cv2
import numpy as np
import datetime
import threading
from queue import Queue



cap = cv2.VideoCapture('D:/BTP/CODE/Videos/120rpm.mp4')

# tracker = cv2.TrackerMOSSE_create()
# tracker = cv2.legacy.TrackerMOSSE_create()
# tracker = cv2.legacy.TrackerCSRT_create()
tracker = cv2.TrackerKCF_create()


drawing = False
start_point = (-1, -1)
end_point = (1, 1)
getPoint = False
prev = 0
now = 0


def line(x,y):
    if (( ((start_point[1]- end_point[1])/(start_point[0]-end_point[0]))*(x-start_point[0])+start_point[1]-y ))>=0:
        return 1
    return 0

# print("line ", start_point, end_point, line(0,0))


# Mouse callback function
def draw_line(event, x, y, flags, param):
    global drawing, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        # return False
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        cv2.imshow("Line", frame)
        getPoint = True
        
    


# Read the first frame
ret, frame = cap.read()

# Create a window and bind the mouse callback function
cv2.namedWindow("Line")
cv2.setMouseCallback("Line", draw_line)
# frame = cv2.resize(frame, (720, 720))
cv2.imshow("Line", frame)
while ret:
    if cv2.waitKey(1) == 13:
        cv2.destroyWindow("Line")
        break
        




success , frame = cap.read()
# frame = cv2.resize(frame, (720, 720))
bbox = cv2.selectROI("Tracking", frame,False)
tracker.init(frame, bbox)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps


tstamp = 0

rev = 0
rTime = 0
def drawBox(frame, bbox):
    # x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # cv2.rectangle(frame, (x,y), ((x+w), (y+h)), (255,0,255), 3, 1)
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, "Tracking", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255, 0), 2)
    cv2.circle(frame, (int(x+w/2),int(y+h/2)), 5, (0,255,0), 1)
    now = line(x,y)
    global prev
    global rev
    global tstamp
    global rTime
    if(prev!= now and tstamp/1000 >= 1):
        print("time " ,tstamp/1000)
        print("ptime ", rTime/1000)
        prev = now
        rev += 1
        print("rev ",int(rev/2))
        
        print("rpm ", 60*1000*(1/((tstamp-rTime)*2)))
        rTime = tstamp


def buffer_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_buffer.put(frame)



def process_frames():
     while True:
    # print(start_point)
    # print(end_point)
    # timer = cv2.getTickCount()
    # success , frame = cap.read()
    frame = video_buffer.get()
    if frame is None:
        break
    
    
    # frame = cv2.resize(frame, (720, 720))

    if not success:
        break



    success, bbox = tracker.update(frame)
    if success:
        drawBox(frame, bbox)
    else:
        cv2.putText(frame, "Lost", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    tstamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    cv2.putText(frame, str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, str(int(tstamp/1000)), (75,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

buffer_thread = threading.Thread(target=buffer_frames)
buffer_thread.start()

# Wait for some time to ensure frames are buffered
# You can adjust this based on your requirements
buffer_thread.join(timeout=2.0)

# Start the frame processing thread
process_thread = threading.Thread(target=process_frames)
process_thread.start()

# Wait for the processing thread to finish
process_thread.join()

# print("rev ",rev)

cap.release()
cv2.destroyAllWindows()
   