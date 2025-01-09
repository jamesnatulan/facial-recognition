import cv2
from ultralytics import YOLO
from PIL import Image
from scripts.common import SiameseResNet
import torch
import torchvision.transforms as transforms
import os
import random

ARCH = "resnet18"
MODEL_PATH = "runs/run2/final.pt"
FACES_PATH = "assets/faces"


def generate_random_color():
    """
    Generate a random color
    """
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def get_nearest_face(model: SiameseResNet, input_face, embeddings):
    """
    Function to get the nearest face from the dataset
    Uses euclidean distance to compare the embeddings
    """
    input_embedding = model.forward_once(input_face)
    min_distance = float("inf")
    nearest_face = None
    for name, emb in embeddings:
        distance = torch.nn.functional.pairwise_distance(
            input_embedding, emb, keepdim=True
        )
        if distance.item() < min_distance:
            min_distance = distance
            nearest_face = name
    return nearest_face


def main():
    print("Hello from facial-recognition!")

    # Load siamese model
    siamese_model = SiameseResNet(ARCH)
    siamese_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    siamese_model.cuda()

    transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # To avoid processing the embeddings for each face in the dataset, embeddings are precomputed
    # and stored in a file. The file is a dictionary with the key being the name of the person and
    # the value being the embedding.
    embs_path = os.path.join(FACES_PATH, "face_embeddings.pt")
    if os.path.exists(embs_path):
        embeddings = torch.load(embs_path)
    else:
        # Precompute face embeddings
        embeddings = []
        for person in os.listdir(FACES_PATH):
            person_path = os.path.join(FACES_PATH, person)
            if os.path.isdir(person_path):
                for face in os.listdir(person_path):
                    face_path = os.path.join(person_path, face)
                    face_img = Image.open(face_path)
                    face_tensor = transform(face_img).unsqueeze(0).cuda()
                    embedding = siamese_model.forward_once(face_tensor)
                    embeddings.append((person, embedding))

        # Save the embeddings
        embs_path = os.path.join(FACES_PATH, "face_embeddings.pt")
        torch.save(embeddings, embs_path)

    # YOLO model for face detection
    model = YOLO(model="models/yolov8n-face.pt")

    # Cam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Assign different colors for each face
    colors = {}
    for person in os.listdir(FACES_PATH):
        colors[person] = generate_random_color()

    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True, verbose=False)

        # Iterate over each detected face
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )

                # Crop face
                face = frame[y1:y2, x1:x2]
                face = Image.fromarray(face)
                face = transform(face).unsqueeze(0).cuda()

                # Get the nearest face
                name = get_nearest_face(siamese_model, face, embeddings)

                # Label the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[name], 3)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[name], 2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()