import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict


class ObjectDetector:

    def __init__(self, model: str):
        self.write_video = False
        self.model = YOLO(model)

        self.names = self.model.model.names

    @staticmethod
    def get_capture_config(capture):
        return (int(capture.get(x)) for x in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS))

    def process(self, data):
        results = self.model.track(data, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:

            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Annotator Init
            annotator = Annotator(data, line_width=2)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

            return data

    def process_camera(self, camera_id):
        cap = cv2.VideoCapture(camera_id)
        assert cap.isOpened(), "Error camera reading"
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                self.process(frame)
                yield frame
            else:
                break
        cap.release()

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = self.get_capture_config(cap)
        result = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps,
                                 (w, h))
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                self.process(frame)
                result.write(frame)
            else:
                break
        result.release()
        cap.release()

    def process_image(self, input_path, output_path):
        image = cv2.imread(input_path)
        self.process(image)
        cv2.imwrite(output_path, image)
