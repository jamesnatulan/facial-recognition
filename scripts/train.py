# Contains the training loop for the siamese network
import torch
from torch.utils.tensorboard import SummaryWriter
from common import SiameseResNet, ContrastiveLoss
from torchvision.datasets import LFWPairs
import torchvision.transforms as transforms
import os
from tqdm import tqdm

ARCH = "resnet18"
DATASET_PATH = "datasets/lfw-deepfunneled"
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
BASE_OUTPUT_DIR = "runs"
EVAL_FREQ = 5
SAVE_FREQ = 5


# Train Function
def train():
    # Setup output
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    run_number = len(os.listdir(BASE_OUTPUT_DIR))
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"run{run_number}")
    os.makedirs(output_dir, exist_ok=True)

    # Load lfw dataset
    transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = LFWPairs(
        root="datasets/lfw-deepfunneled",
        transform=transform,
        image_set="deepfunneled",
        download=True,
        split="train",
    )
    eval_dataset = LFWPairs(
        root="datasets/lfw-deepfunneled",
        transform=transform,
        image_set="deepfunneled",
        download=True,
        split="test",
    )

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    print("Train Samples: ", len(train_dataset))
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    print("Eval Samples: ", len(eval_dataset))

    # Initialize the model, loss function, and optimizer
    model = SiameseResNet(ARCH).cuda()
    loss_fn = ContrastiveLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005
    )

    # Initialize Tensorboard writer
    log_dir = os.path.join(output_dir, "logs")
    writer = SummaryWriter(log_dir)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train Loop

        # Set model to training mode
        model.train()
        train_loss = 0
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()

            img1, img2, label = batch
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()

            # Forward pass
            output1, output2 = model(img1, img2)
            loss_contrastive = loss_fn(output1, output2, label)

            # Backward pass
            loss_contrastive.backward()
            optimizer.step()

            train_loss += loss_contrastive.item()
            pbar.set_description(
                f"Epoch {epoch}/{NUM_EPOCHS}, Train Loss: {loss_contrastive.item():.4f}"
            )

        train_loss /= len(train_dataloader)
        pbar.set_description(
            f"Epoch {epoch}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}"
        )
        writer.add_scalar(
            "Train Loss",
            train_loss,
            global_step=epoch,
        )

        # Eval Loop
        # Evaluate the model every EVAL_FREQ epochs
        if epoch % EVAL_FREQ == 0:
            # Set model to evaluation mode
            model.eval()
            eval_loss = 0
            pbar = tqdm(eval_dataloader, total=len(eval_dataloader))
            for step, batch in enumerate(pbar):
                with torch.no_grad():
                    img1, img2, label = batch
                    img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()

                    # Forward pass
                    output1, output2 = model(img1, img2)
                    loss_contrastive = loss_fn(output1, output2, label)

                eval_loss += loss_contrastive.item()
                pbar.set_description(
                    f"Epoch {epoch}/{NUM_EPOCHS}, Validation Loss: {loss_contrastive.item():.4f}"
                )

            eval_loss /= len(eval_dataloader)
            pbar.set_description(
                f"Epoch {epoch}/{NUM_EPOCHS}, Validation Loss: {eval_loss:.4f}"
            )
            writer.add_scalar(
                "Validation Loss",
                eval_loss,
                global_step=epoch,
            )

            writer.add_scalars(
                "Loss",
                {"training": train_loss, "validation": eval_loss},
                global_step=epoch,
            )

        # Save a checkpoint every SAVE_FREQ epochs
        if epoch % SAVE_FREQ == 0:
            # Save model
            save_path = os.path.join(output_dir, f"checkpoint_{epoch}.pt")
            torch.save(model.state_dict(), save_path)

    # Close the writer
    writer.close()

    # Save model
    save_path = os.path.join(output_dir, "final.pt")
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()
