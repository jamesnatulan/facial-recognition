# Contains the training loop for the siamese network
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import SiameseDataset
from model import SiameseNetwork
from loss import ContrastiveLoss

import os
from tqdm import tqdm


DATASET_PATH = "datasets/lfw-yolo"
NUM_EPOCHS = 50
BATCH_SIZE = 16
BASE_OUTPUT_DIR = "runs"

# Train Function
def train():
    # Setup output 
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    run_number = len(os.listdir(BASE_OUTPUT_DIR))
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"run{run_number}")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    image_pairs_path = os.path.join(DATASET_PATH, "image_pairs.csv")
    dataset = SiameseDataset(image_pairs_path)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8
    )

    # Initialize the model, loss function, and optimizer
    model = SiameseNetwork().cuda()
    loss_fn = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)

     # Initialize Tensorboard writer
    log_dir = os.path.join(output_dir, "logs")
    writer = SummaryWriter(log_dir)
        
    # Set model to training mode    
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
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
            f"Epoch {epoch}/{NUM_EPOCHS}, Train Loss: {train_loss.item():.4f}"
        )
        writer.add_scalar(
            "Train Loss",
            train_loss,
            global_step=epoch,
        )
    
    # Close the writer
    writer.close()

    # Save model
    save_path = os.path.join(output_dir, "siamese_network.pt")
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()