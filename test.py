from model import SiameseResNet
from PIL import Image
import torch
import torchvision.transforms as transforms

def main():
    siamese_model = SiameseResNet("resnet18")
    siamese_model.load_state_dict(torch.load("runs/run1/final.pt", weights_only=True))
    siamese_model.cuda()

    # Faces
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    face_1_path = "assets/tests/test_face_6.jpg"
    face_1 = Image.open(face_1_path)
    face_1 = transform(face_1).unsqueeze(0)

    face_2_path = "assets/tests/test_face_7.jpg"
    face_2 = Image.open(face_2_path)
    face_2 = transform(face_2).unsqueeze(0)

    with torch.no_grad():
        face_1 = face_1.cuda()
        face_2 = face_2.cuda()
        output1, output2 = siamese_model(face_1, face_2)
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        print(euclidean_distance.item())
        if euclidean_distance.item() < 0.5:
            print("Match!")

if __name__ == "__main__":
    main()
