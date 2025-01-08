import cv2
from ultralytics import YOLO
from model import SiameseResNet
from PIL import Image
import torch
import torchvision.transforms as transforms

def main():
    print("Hello from facial-recognition!")

    model = YOLO(model='models/yolov8n-face.pt')
    siamese_model = SiameseResNet("resnet18")
    siamese_model.load_state_dict(torch.load("runs/run1/final.pt"))
    siamese_model.cuda()

    # Cam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Base face
    base_face_path = "assets/tests/test_face_3.jpg"
    base_face = Image.open(base_face_path)
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    base_face = transform(base_face).unsqueeze(0)

    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True, verbose=False)
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # crop face
                face = frame[y1:y2, x1:x2]
                face = Image.fromarray(face)
                face = transform(face).unsqueeze(0)

                # compare faces
                with torch.no_grad():
                    base_face = base_face.cuda()
                    face = face.cuda()
                    output1, output2 = siamese_model(base_face, face)
                    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
                    print(euclidean_distance.item())
                    if euclidean_distance.item() < 0.5:
                        print("Match!")

                # rect in frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
