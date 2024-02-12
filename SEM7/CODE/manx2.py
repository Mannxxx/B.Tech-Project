import cv2

# Global variables for mouse callback
drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

# Function to zoom and process a frame
def process_frame(frame, zoom_factor):
    global start_x, start_y, end_x, end_y
    
    # Calculate the zoomed region
    x_zoom = max(0, start_x - int((end_x - start_x) * zoom_factor))
    y_zoom = max(0, start_y - int((end_y - start_y) * zoom_factor))
    w_zoomed = min(frame.shape[1], end_x + int((end_x - start_x) * zoom_factor)) - x_zoom
    h_zoomed = min(frame.shape[0], end_y + int((end_y - start_y) * zoom_factor)) - y_zoom
    
    # Zoom and crop the frame
    zoomed_frame = frame[y_zoom:y_zoom + h_zoomed, x_zoom:x_zoom + w_zoomed]
    
    return zoomed_frame

# Define the input and output video paths
input_video_path = 'D:/BTP/CODE/Videos/video_1.mp4'
output_video_path = 'D:/BTP/CODE/Videos/output_1.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define zoom parameter
zoom_factor = 0.1  # Adjust this value to control the amount of zoom

# Create a window and set mouse callback
cv2.namedWindow('Select Zoom Area')
cv2.setMouseCallback('Select Zoom Area', draw_rectangle)

# Define the codec and create VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame
    if end_x != -1 and end_y != -1:
        zoomed_frame = process_frame(frame, zoom_factor)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  # Draw the selected area
        
        # Write the processed frame to the output video
        out.write(zoomed_frame)
        
    cv2.imshow('Select Zoom Area', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
