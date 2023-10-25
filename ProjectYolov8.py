from ultralytics import YOLO
import cv2

# import ultralytics
# ultralytics.checks()

#load yolov8 model
model = YOLO('best.pt')

# load video
video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)


# Create a VideoWriter object
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), fps,(height, width))


# read frames
for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, frame = cap.read()
    
    # detect object
    # track object
    results = model.track(frame, persist=True)
        
    # plot results
    frame_ = results[0].plot()

    
    
    # visualize
    cv2.imshow('frame',frame_)
    out.write(frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()