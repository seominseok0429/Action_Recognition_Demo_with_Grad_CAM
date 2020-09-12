import torch
import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
NAME = 0
LABEL = ['Brush_hair' , 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble',
         'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac', 'golf', 'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball',
         'kiss', 'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup', 'ride_bike', 'ride_horse', 'run', 'shake_hands',
         'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault', 'stand', 'swing_baseball',
         'sword', 'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave']


def get_video_frame(args):
    capture = cv2.VideoCapture(args.video_path)
    start_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stack_frame = []
    for frame in range(start_frame, end_frame-1):
        ret, fram = capture.read()
        fram = cv2.resize(fram, dsize=(112, 112), interpolation=cv2.INTER_AREA)
        fram = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
        stack_frame.append(fram)
    stack_frame = np.asarray(stack_frame, dtype=np.float32)
    return torch.from_numpy(stack_frame.transpose([3,0,1,2])).unsqueeze(0), end_frame


def show_cam_on_image(img, mask, frame, pred5, pred_time, args):
    global NAME
    model_name = '3D-ResNeXt101'
    for i in range(0,64):
        canvers = np.zeros((130,480,3), np.uint8)
        canvers[:,:,] = 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask[:,:,i]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)
        dst = cv2.addWeighted(img[i,:,:,:], 0.5, heatmap, 0.5, 0)
        dst = cv2.resize(dst, dsize=(240, 240), interpolation=cv2.INTER_AREA)
        img1 = img[i,:,:,:]
        img1 = cv2.resize(img1, dsize=(240, 240), interpolation=cv2.INTER_AREA)
        #icv2.imwrite('./grad_cam/'+str(NAME)+"cam.jpg", dst)
        pred1 = 'prad : ' + LABEL[pred5[0]]
        ptime = ',  time : ' + str(pred_time)[:7]
        cv2.putText(img1, pred1, (3, 10), font, 0.3, (0, 0, 0), 1)
        cv2.putText(img1, ptime, (90, 10), font, 0.3, (0, 0, 0), 1)
        #cv2.imwrite('./grad_cam/'+str(NAME)+"rgb.jpg", img1)
        stack_img = np.hstack((img1, dst))
        stack_img = np.vstack((stack_img,canvers))
        pred1 = 'prad1 : ' + LABEL[0]
        pred2 = 'prad2 : ' + LABEL[pred5[1]]
        pred3 = 'prad3 : ' + LABEL[pred5[2]]
        pred4 = 'prad4 : ' + LABEL[pred5[3]]
        pred6 = 'prad5 : ' + LABEL[pred5[4]]
        cv2.putText(stack_img, model_name, (3, 250), font, 0.3, (0, 0, 0), 1)
        cv2.putText(stack_img, pred1, (3, 270), font, 0.3, (0, 0, 0), 1)
        cv2.putText(stack_img, pred2, (3, 290), font, 0.3, (0, 0, 0), 1)
        cv2.putText(stack_img, pred3, (3, 310), font, 0.3, (0, 0, 0), 1)
        cv2.putText(stack_img, pred4, (3, 330), font, 0.3, (0, 0, 0), 1)
        cv2.putText(stack_img, pred6, (3, 350), font, 0.3, (0, 0, 0), 1)
        path = './final/'+'%05d.jpg'%NAME
        cv2.imwrite(path, stack_img)
        show_img = cv2.imread(path)
        cv2.imshow('hi', show_img)
        cv2.waitKey(15)
        NAME +=1

