import random
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from files import increment_path


def main(exist_ok=True, view_img=False, save_img=True):
    source = 'video.mp4'
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    yolov8_model_path = 'models/yolov8n.pt'
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                         model_path=yolov8_model_path,
                                                         confidence_threshold=0.3,
                                                         device='cpu')

    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, fourcc = 30, cv2.VideoWriter_fourcc(*'mp4v')
    save_dir = increment_path(Path('output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))

    while videocapture.isOpened():
        success, frame = videocapture.read()

        if not success:
            break

        results = get_sliced_prediction(frame,
                                        detection_model,
                                        slice_height=512,
                                        slice_width=512,
                                        overlap_height_ratio=0.2,
                                        overlap_width_ratio=0.2)
        object_prediction_list = results.object_prediction_list

        boxes_list = []
        clss_list = []
        for ind, _ in enumerate(object_prediction_list):
            boxes = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
                object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
            clss = object_prediction_list[ind].category.name
            boxes_list.append(boxes)
            clss_list.append(clss)

        for box, cls in zip(boxes_list, clss_list):
            random_rgb = [random.randint(0, 255) for _ in range(3)]
            label = str(cls)
            x1, y1, x2, y2 = box

            label_colors = {
                'person': (0, 100, 0),
                'car': (160, 82, 45),
                'bus': (138, 43, 226),
                'truck': (0, 100, 0),
                'backpack': (255, 105, 180),
                'handbag': (255, 105, 180),
                'dog': (144, 238, 144),
                'traffic light': (139, 0, 0),
                'umbrella': (240, 230, 140),
                'baseball glove': (240, 230, 140)
            }

            color = label_colors.get(label, random_rgb)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # label
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
            cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), color,
                          -1)
            cv2.putText(frame,
                        label, (int(x1), int(y1) - 2),
                        0,
                        0.6, [255, 255, 255],
                        thickness=1,
                        lineType=cv2.LINE_AA)

            if view_img:
                cv2.imshow(Path(source).stem, frame)
            if save_img:
                video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_writer.release()
        videocapture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
