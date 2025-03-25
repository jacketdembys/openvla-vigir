
#export QT_QPA_PLATFORM=offscreen


import cv2
import matplotlib.pyplot as plt


width = 1280# 224
height = 960 #224
            

def stream_video():
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # plt.ion()  # Turn on interactive mode
    # fig, ax = plt.subplots()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        cv2.imshow("frame", frame)
        # cv2.imwrite(f"frame_{frame_count}.png", frame)
        frame_count+=1
        # if frame_count > 10:
        #    break
        # print(frame.shape)
        # print(frame)
        # if 'im' in locals():
        #     im.set_data(frame)
        # else:
        #     im = ax.imshow(frame)
        # plt.pause(0.01)
        # plt.draw()

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    #plt.close()

if __name__ == '__main__':
    stream_video()
