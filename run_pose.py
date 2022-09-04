import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
from trainer import findAngle


@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu', biceptrainer = False, drawkeypoints = True):

    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        vid_write_image = letterbox(
            cap.read()[1], (frame_width), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_result4.mp4", cv2.VideoWriter_fourcc(
            *'mp4v'), 30, (resize_width, resize_height))

        frame_count, total_fps = 0, 0

        while cap.isOpened:

            print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                orig_image = frame

                # preprocess image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (frame_width),
                                  stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(
                    output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if biceptrainer:
                    for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        # Right arm =(5,7,9), left arm = (6,8,10)
                        angle = findAngle(img, kpts, 5, 7, 9, draw=False)
                        percentage = np.interp(angle, (210, 290), (0, 100))
                        bar = np.interp(angle, (220, 290), (650, 100))
        

                        color = (53, 14, 133)
                        # check for the bicep curls
                        if percentage == 100:
                            color = (0, 255, 0)
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                        if percentage == 0:
                            color = (0, 255, 0)
                            if direction == 1:
                                bcount += 0.5
                                direction = 0

                        # draw Bar
                        cv2.rectangle(img, (50, 100), (100, 650), color, 3)
                        cv2.rectangle(img, (50, int(bar)), (100, 650), color, -1)
                        cv2.putText(img, f"{int(percentage)}%", (50, 80),
                                    cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

                        # draw bicep curl
                        cv2.rectangle(img, (0, 700), (250, 1088), (228, 245, 255), -1)
                        cv2.putText(img, f"{int(bcount)}", (45, 970),
                                    cv2.FONT_HERSHEY_PLAIN, 15, (53, 14, 133), 35)

                if drawkeypoints:
                    for idx in range(output.shape[0]):
                        plot_skeleton_kpts(img, output[idx, 7:].T, 3)

                if ext.isnumeric():
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img)
            else:
                break

        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str,
                        default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str,
                        help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--biceptrainer', type=bool, default=False,
                        help='set as true to check count bicep curls')
    parser.add_argument('--drawkeypoints', type=bool,
                        default=True, help='draw all keypoints')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
