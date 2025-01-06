import cv2
from ultralytics import YOLO

def main():
    print("Hello from facial-recognition!")

    model = YOLO(model='models/yolov8n-face.pt')

    # Cam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True)
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # rect in frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
