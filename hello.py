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
    siamese_model.load_state_dict(torch.load("runs/run21/final.pt"))
    siamese_model.cuda()

    # Cam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Base faces
    faces = [
        ("Angelina Jolie", "assets/tests/test_face_1.jpg"),
        ("Brad Pitt", "assets/tests/test_face_2.jpg"),
        ("Kua J", "assets/tests/test_face_3.jpg"),
    ]
    faces_embs = []
    for name, face in faces:
        face = Image.open(face)
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
        face = transform(face).unsqueeze(0)
        faces_embs.append((name, face))

    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True, verbose=False)
        bbox_color = (0, 0, 255)
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
                distances = []
                with torch.no_grad():
                    for name, base_face in faces_embs:
                        base_face = base_face.cuda()
                        face = face.cuda()
                        output1, output2 = siamese_model(base_face, face)
                        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
                        distances.append((name, euclidean_distance.item()))

                # Get lowest distance
                distances = sorted(distances, key=lambda x: x[1])
                name, distance = distances[0]

                # draw text and change bbox color only if distance is less than 0.5
                # and face matches kua J
                if distance < 0.5 and name == "Kua J":
                    bbox_color = (0, 255, 0)
                    cv2.putText(frame, f"{name} - {distance:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bbox_color, 2)

                # rect in frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 3)
                   

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
