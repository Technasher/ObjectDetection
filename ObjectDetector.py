import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class ObjectDetector:

    def __init__(self, model: str):
        self.model = YOLO(model)

    @staticmethod
    def get_capture_config(capture):
        return (int(capture.get(x)) for x in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS))

    def process(self, data):
        results = self.model.track(data, verbose=False)[0]

        if results.boxes.id is not None:

            boxes = results.boxes.xyxy.cpu()
            clss = results.boxes.cls.cpu().tolist()

            annotator = Annotator(data, line_width=2)

            for box, cls in zip(boxes, clss):
                annotator.box_label(box, color=colors(int(cls), True), label=self.model.model.names[int(cls)])

    def process_camera(self, camera_id):
        cap = cv2.VideoCapture(camera_id)
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
