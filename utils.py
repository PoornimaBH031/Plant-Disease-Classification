# utils.py

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.manifold import TSNE

# ==========================================================
# AVERAGE CLASSIFICATION RATE (ACR)
# ==========================================================

def calculate_acr(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    class_acc = []

    for i in range(len(cm)):

        total = np.sum(cm[i])

        if total > 0:

            acc = (cm[i, i] / total) * 100

            class_acc.append(acc)

    acr = np.mean(class_acc)

    return acr

# ==========================================================
# SAVE CHECKPOINT
# ==========================================================

def save_checkpoint(
        model,
        path):

    torch.save(
        model.state_dict(),
        path
    )

    print(
        f"Checkpoint Saved: {path}"
    )


# ==========================================================
# LOAD CHECKPOINT
# ==========================================================

def load_checkpoint(
        model,
        path,
        device):

    model.load_state_dict(
        torch.load(
            path,
            map_location=device
        )
    )

    model.eval()

    print(
        f"Checkpoint Loaded: {path}"
    )

    return model


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

def print_classification_report(
        y_true,
        y_pred,
        class_names):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print(report)

    return report


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path=None):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=45
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300
        )

    plt.show()


# ==========================================================
# TSNE VISUALIZATION
# ==========================================================

def plot_tsne(
        features,
        labels,
        save_path=None):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    reduced = tsne.fit_transform(
        features
    )

    plt.figure(
        figsize=(8, 8)
    )

    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10"
    )

    plt.legend(
        *scatter.legend_elements(),
        title="Classes"
    )

    plt.title(
        "t-SNE Feature Distribution"
    )

    if save_path:
        plt.savefig(
            save_path,
            dpi=300
        )

    plt.show()


# ==========================================================
# LEARNING CURVES
# ==========================================================

def plot_learning_curves(
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        save_path=None):

    epochs = range(
        1,
        len(train_loss) + 1
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Val Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.title(
        "Loss Curve"
    )

    if save_path:
        plt.savefig(
            os.path.join(
                save_path,
                "loss_curve.png"
            ),
            dpi=300
        )

    plt.show()

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        train_acc,
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_acc,
        label="Val Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.title(
        "Accuracy Curve"
    )

    if save_path:
        plt.savefig(
            os.path.join(
                save_path,
                "accuracy_curve.png"
            ),
            dpi=300
        )

    plt.show()


# ==========================================================
# FEATURE EXTRACTION FOR TSNE
# ==========================================================

def extract_features(
        model,
        dataloader,
        device):

    model.eval()

    features = []

    labels = []

    with torch.no_grad():

        for images, target in dataloader:

            images = images.to(device)

            output = model(images)

            features.append(
                output.cpu().numpy()
            )

            labels.append(
                target.numpy()
            )

    features = np.concatenate(
        features,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    return features, labels


# ==========================================================
# GRAD-CAM
# ==========================================================

class GradCAM:

    def __init__(
            self,
            model,
            target_layer):

        self.model = model

        self.target_layer = target_layer

        self.gradients = None

        self.activations = None

        self.hook_layers()

    def hook_layers(self):

        def forward_hook(
                module,
                inp,
                out):

            self.activations = out

        def backward_hook(
                module,
                grad_in,
                grad_out):

            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def generate(
            self,
            image_tensor,
            class_idx=None):

        output = self.model(
            image_tensor
        )

        if class_idx is None:

            class_idx = torch.argmax(
                output
            ).item()

        self.model.zero_grad()

        output[:, class_idx].backward()

        gradients = self.gradients

        activations = self.activations

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(
            weights * activations,
            dim=1
        )

        cam = torch.relu(cam)

        cam = cam.squeeze()

        cam = cam.cpu().detach().numpy()

        cam = cv2.resize(
            cam,
            (128, 128)
        )

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return cam


# ==========================================================
# OVERLAY HEATMAP
# ==========================================================

def overlay_gradcam(
        image,
        cam):

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = np.float32(
        heatmap
    ) / 255

    image = np.float32(
        image
    ) / 255

    overlay = (
        heatmap * 0.4 +
        image * 0.6
    )

    overlay = np.uint8(
        overlay * 255
    )

    return overlay