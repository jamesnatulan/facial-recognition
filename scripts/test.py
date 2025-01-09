from tqdm import tqdm
from common import SiameseResNet
import torch
import torchvision.transforms as transforms
from torchvision.datasets import LFWPairs

ARCH = "resnet18"
MODEL_PATH = "runs/run2/final.pt"


def main():
    model = SiameseResNet(ARCH)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.cuda()
    model.eval()

    # Load test dataset
    transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = LFWPairs(
        root="datasets/lfw-deepfunneled",
        transform=transform,
        image_set="deepfunneled",
        download=True,
        split="test",
    )
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    tp = 0 # Label is 1, prediction is 1
    tn = 0 # Label is 0, prediction is 0
    fp = 0 # Label is 0, prediction is 1
    fn = 0 # Label is 1, prediction is 0

    pbar = tqdm(dataloader, total=len(dataloader))
    for step, batch in enumerate(pbar):
        with torch.no_grad():
            img1, img2, label = batch
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()

            # Forward pass
            output1, output2 = model(img1, img2)
        
        # Calculate euclidean distance
        euclidean_distance = torch.nn.functional.pairwise_distance(
            output1, output2, keepdim=True
        )
        euclidean_distance = euclidean_distance.item()
        prediction = 1 if euclidean_distance < 0.5 else 0

        if label.item() == 1 and prediction == 1:
            tp += 1
        elif label.item() == 0 and prediction == 0:
            tn += 1
        elif label.item() == 0 and prediction == 1:
            fp += 1
        elif label.item() == 1 and prediction == 0:
            fn += 1

    # Calcualte accuracy and precision
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
            


if __name__ == "__main__":
    main()
