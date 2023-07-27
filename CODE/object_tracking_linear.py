import cv2
import numpy as np
import datetime


cap = cv2.VideoCapture("C:/Users/Mansi/Desktop/BTP/CODE/Videos/speed2.mp4")

video_type = int(input('Select video type: \n 1. Linear \n 2. Rotational \n'))

# tracker = cv2.TrackerMOSSE_create()
# tracker = cv2.legacy.TrackerMOSSE_create()
tracker = cv2.legacy.TrackerCSRT_create()

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
def set_CSRT_Params():
    # Don't modify
    default_params = {
        'padding': 3.,
        'template_size': 200.,
        'gsl_sigma': 1.,
        'hog_orientations': 9.,
        'num_hog_channels_used': 18,
        'hog_clip': 2.0000000298023224e-01,
        'use_hog': 1,
        'use_color_names': 1,
        'use_gray': 1,
        'use_rgb': 0,
        'window_function': 'hann',
        'kaiser_alpha': 3.7500000000000000e+00,
        'cheb_attenuation': 45.,
        'filter_lr': 1.9999999552965164e-02,
        'admm_iterations': 4,
        'number_of_scales': 100,
        'scale_sigma_factor': 0.25,
        'scale_model_max_area': 512.,
        'scale_lr': 2.5000000372529030e-02,
        'scale_step': 1.02,
        'use_channel_weights': 1,
        'weights_lr': 1.9999999552965164e-02,
        'use_segmentation': 1,
        'histogram_bins': 16,
        'background_ratio': 2,
        'histogram_lr': 3.9999999105930328e-02,
        'psr_threshold': 3.5000000149011612e-02,
    }
    # modify
    params = {
        'scale_lr': 0.5,
        'number_of_scales': 250,
        'hog_orientations': 100,
        'use_segmentation': 100,
        'admm_iterations': 1000,
        'hog_clip' : 2,
        'background_ratio':1
    }
    params = {**default_params, **params}
    tracker = None
    if int(major_ver) == 3 and 3 <= int(minor_ver) <= 4:
        import json
        import os
        with open('tmp.json', 'w') as fid:
            json.dump(params, fid)
        fs_settings = cv2.FileStorage("tmp.json", cv2.FILE_STORAGE_READ)
        tracker = cv2.TrackerCSRT_create()
        tracker.read(fs_settings.root())
        os.remove('tmp.json')
    elif int(major_ver) >= 4:
        param_handler = cv2.TrackerCSRT_Params()
        for key, val in params.items():
            setattr(param_handler, key, val)
        tracker = cv2.TrackerCSRT_create(param_handler)
    else:
        print("Cannot set parameters, using defaults")
        tracker = cv2.TrackerCSRT_create()
    return tracker

tracker = set_CSRT_Params()

points = []  #stores (drawing, start_point, end_point) of every line
prev_turn = [] #previously where was tracker w.r.t ith line
time_change = [] #time history of every line when tracker cross that line 
avg_val = [] #speed or rpm stored



def line(x,y):
    global points
    global prev_turn
    nturn = []
    for p in points:
        if (( (p[1][1]- p[2][1])*(x-p[1][0])+(p[1][1]-y)*(p[1][0]-p[2][0]) ))>=0:
            nturn.append(1)
        else:
            nturn.append(0)
    if(len(prev_turn)== 0):
        prev_turn = nturn
        return prev_turn
    
    changes = [prev_turn[i] ^ nturn[i] for i in range(len(nturn))]
    prev_turn = nturn
    return changes
    






# Mouse callback function
def draw_line(event, x, y, flags, param):
    global points
    global time_change
    window = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append([drawing, (x,y), (-1,1)])
    elif event == cv2.EVENT_LBUTTONUP:
        points[-1][0] = False
        # points[-1][2] = (points[-1][1][0], y)

        if(video_type == 1):
            points[-1][2] = (points[-1][1][0], y)
            cv2.line(frame, points[-1][1], points[-1][2], (0, 0, 255), 2)

            if(len(points) == 2):
                cv2.putText(frame, "Press ENTER", (175,175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        if(video_type == 2):
            points[-1][2] = (x, y)
            cv2.line(frame, points[-1][1], points[-1][2], (0, 0, 255), 2)
            cv2.putText(frame, "Press ENTER", (175,175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow(window, frame)
        time_change.append([0,0])
        
    


# Read the initial frame
ret, frame = cap.read()

if(video_type == 1):
    distace_traveled = int(input("distance between two flags:\n"))
    cv2.putText(frame, "Draw two Lines:", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0, 0), 2)
if(video_type == 2):
    cv2.putText(frame, "Draw one straight line through center:", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

# Create a window and bind the mouse callback function
cv2.namedWindow("Line")
cv2.setMouseCallback("Line", draw_line, "Line")
frame_height, frame_width = frame.shape[:2]
# frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2)))



cv2.imshow("Line", frame)


while ret:
    if cv2.waitKey(1) & 0xff == 13:
        cv2.destroyWindow("Line")
        break




success , frame = cap.read()
# frame = cv2.resize(frame, (720, 720))
# frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2)))



cv2.putText(frame, "Draw tracker on the Mark and then press ENTER:", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
bbox = cv2.selectROI("Tracking", frame,False)
tracker.init(frame, bbox)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps


tstamp = 0
rev = 0
rTime = 0

def drawBox(frame, bbox):
    global rev
    global tstamp
    global rTime
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x,y), ((x+w), (y+h)), (255,0,255), 3, 1)
    cv2.putText(frame, "Tracking", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255, 0), 2)
    cv2.circle(frame, (int(x+w/2),int(y+h/2)), 5, (0,255,0), 1)
    
    
    changes = line(x+w/2,y+h/2)


    for i in range(len(changes)):
        if(changes[i] == 1):
            # if(len(time_change[i])>0):
            time_dif = tstamp/1000 -time_change[i][-1]
            # print(time_dif)
            if(time_dif > 0):
                if(video_type == 1):
                    avg_val.append(distace_traveled/(tstamp/1000-time_change[i-1][-1]))
                if(video_type == 2):
                    avg_val.append(60*(1/((time_dif)*2)))

            time_change[i][0] = time_change[i][1]
            time_change[i][1] = (tstamp/1000)
    # print(time_change)



while True:

    success , frame = cap.read()

    if not success:
        break


    # frame = cv2.resize(frame, (720, 720))
    # frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2)))
    # tracker.init(frame, params)

    success, bbox = tracker.update(frame)
    if success:
        drawBox(frame, bbox)
    else:
        cv2.putText(frame, "Lost", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    tstamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    cv2.putText(frame, str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, str(int(tstamp/1000)), (75,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    
    for p in points:
        cv2.line(frame, p[1], p[2], (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        # print(tstamp/1000)
        # continue



# print(time_change[1][-1]- time_change[0][-1])
# print(distace_traveled/(time_change[1][-1]- time_change[0][-1]))
print(avg_val)
avg_val.pop(0)
if(video_type == 2):
    print("Average rpm is:\n")
if(video_type == 1):
    print("Average Speed is:\n")

print(sum(avg_val)/len(avg_val))
cap.release()
cv2.destroyAllWindows()
   