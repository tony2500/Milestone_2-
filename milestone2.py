import torch
import cv2
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

images = []
img_names = []
videos = []
vid_names = []


def sel_img():
    file = filedialog.askopenfilename(title='choose images', multiple=True, filetype=(
        ('.png files', '*.png'), ('.jpg files', '*.jpg'), ('all files', '*.*')))
    for i in file:
        images.append(cv2.imread(i))
        path = i[:-4]
        img_names.append(path)
    for i in range(len(images)):
        img_0 = images[i]
        name_0 = img_names[i]
        output = score_frame(img_0)
        res = plot_boxes(output, img_0)
        cv2.imwrite(name_0 + '_detection.png', res)
    messagebox.showinfo('Done', 'Detection process saved, please check the image selected directory')


def sel_vid():
    file = filedialog.askopenfilename(title='choose images', multiple=True, filetype=(
        ('.mp4 files', '*.mp4'), ('.mkv files', '*.mkv'), ('all files', '*.*')))
    for i in file:
        videos.append(cv2.VideoCapture(i))
        path = i[:-4]
        vid_names.append(path)
    for i in range(len(videos)):
        video_0 = videos[i]
        name_0 = vid_names[i]
        frame_width = int(video_0.get(3))
        frame_height = int(video_0.get(4))
        frame_size = (frame_width, frame_height)
        req = cv2.VideoWriter(name_0 + '_detected.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, frame_size)
        # output = cv2.VideoWriter(name_0 + '_detected', cv2.VideoWriter_fourcc(*’MPEG’), 20, frame_size)
        while video_0.isOpened():
            ret, frame = video_0.read()
            if ret:
                output = score_frame(frame)
                res = plot_boxes(output, frame)
                req.write(res)
            else:
                break
    messagebox.showinfo('Done', 'Detection process saved, please check the image selected directory')


def score_frame(frame):
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def plot_boxes(results, frame):
    labels, cord = results
    lmv = ['auto rickshaw', 'bicycle', 'car', 'garbagevan', 'human hauler', 'minivan', 'motorbike', 'policecar',
           'rickshaw', 'scooter', 'suv', 'taxi', 'three wheelers -CNG-', 'wheelbarrow']
    # HMV = ['ambulance', 'army vehicle', 'bus', 'minibus', 'pickup', 'truck', 'van']
    res = []
    for i in labels:
        if classes[int(i)] in lmv:
            res.append('LMV')
        else:
            res.append('HMV')
    n = len(labels)
    HMV = 0
    LMV = 0
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.6:
            if res[i] == 'LMV':
                LMV = LMV + 1
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, res[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            else:
                HMV = HMV + 1
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, res[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    cv2.rectangle(frame, (0, 0), (150, 150), (0, 0, 0), -1)
    cv2.putText(frame, 'HMV' + str(HMV), (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, 'LMV' + str(LMV), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, 'Total' + str(HMV + LMV), (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best__325.pt', force_reload=True)
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)


main_win = Tk()
main_win.title('Object detection')
main_win.geometry('500x500')

Label(main_win, text='Welcome to Image Classifier :)', font="500").place(x=50, y=150)
Label(main_win, text='Please select images: ', font="50").place(x=50, y=250)
Label(main_win, text='Please select video: ', font="50").place(x=50, y=300)

img_btn = Button(main_win, text="Browse..", width=10, height=1, command=sel_img, bg='cyan')
img_btn.place(x=250, y=250)
img_btn = Button(main_win, text="Browse..", width=10, height=1, command=sel_vid, bg='cyan')
img_btn.place(x=250, y=300)

mainloop()
