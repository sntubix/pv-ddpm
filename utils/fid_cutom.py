import torch
import numpy as np
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import PIL.Image as Image
import torchvision.transforms as transforms
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import pacmap
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import lpips
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config
IMAGE_SIZE = (512, 512)
NUM_CLASSES = 30
BATCH_SIZE = 1
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepLabV3 with custom dataset paths")

    # fake image directories
    parser.add_argument("--fake_mono", required=True, help="Path to mono-c-Si fake images")
    parser.add_argument("--fake_multi", required=True, help="Path to multi-c-Si fake images")
    parser.add_argument("--fake_multihalfcut", required=True, help="Path to half-cut multi-c-Si fake images")
    parser.add_argument("--fake_dogbone", required=True, help="Path to IBC-dogbone fake images")

    # real images directory
    parser.add_argument("--real", required=True, help="Path to real images")

    # model weights
    parser.add_argument("--model_weights", required=True, help="Path to trained model weights (.pth file)")

    return parser.parse_args()

args = parse_args()
    
fake_images_mono_path = args.fake_mono
fake_images_multi_path = args.fake_multi
fake_images_multihalfcut_path = args.fake_multihalfcut
fake_images_dogbone_path = args.fake_dogbone

fake_images_paths = {
    "mono-c-Si": fake_images_mono_path,
    "multi-c-Si": fake_images_multi_path,
    "half-cut multi-c-Si": fake_images_multihalfcut_path,
    "IBC-dogbone": fake_images_dogbone_path,
}
real_images_path = args.real

cell_type_colors = {
    "mono-c-Si": {"real": "#1f77b4", "fake": "#aec7e8"},  # Blue tones
    "multi-c-Si": {"real": "#2ca02c", "fake": "#98df8a"},  # Green tones
    "half-cut multi-c-Si": {"real": "#ff7f0e", "fake": "#ffbb78"},  # Orange tones
    "IBC-dogbone": {"real": "#d62728", "fake": "#ff9896"},  # Red tones
}

# Load model with pretrained weights
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
# Replace classifier to match your number of classes (30)
model.classifier[4] = nn.Conv2d(256, 30, kernel_size=1)
# Load your custom-trained model weights (after modifying the classifier!)
model.load_state_dict(torch.load(args.model_weights))
model.eval()

# Wrap encoder
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone
        self._out_feature = 2048 

    def forward(self, x):
        x = self.encoder(x)
        if isinstance(x, dict):
            x = x['out']
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)
    @property
    def num_features(self): 
        return self._out_feature

feature_model = FeatureExtractor(model.backbone).to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # converts to float32 and scales to [0, 1]
    transforms.Normalize([0.5]*3, [0.5]*3)  # normalization for 3 channels
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return transform(img)

all_features = []
all_labels = []
all_types = []

for cell_type, fake_images_path in fake_images_paths.items():
    if not os.path.exists(fake_images_path):
        print(f"Path does not exist: {fake_images_path}")
    fake_images_names = sorted([f for f in os.listdir(fake_images_path) if f.endswith(".png")])
    fake_images = [load_image(os.path.join(fake_images_path, name)) for name in fake_images_names]

    real_images_names = set()
    for image_name in fake_images_names:
        if image_name.endswith(".png"):
            image_name = image_name.split("-")[1]
            parts = image_name.split("_")
            real_image_name = "_".join(parts[:-1]) + ".png"
            real_images_names.add(real_image_name)

    real_images_names = sorted(real_images_names)     
    real_images = [load_image(os.path.join(real_images_path, name)) for name in real_images_names if name.endswith(".png")]
    print(f"{cell_type}: {len(real_images)} real, {len(fake_images)} fake")

    # FID
    fid = FrechetInceptionDistance(feature=feature_model, normalize=True)
    fid = fid.to(DEVICE)
    fid_original = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    # KID
    num_real = len(real_images)
    num_fake = len(fake_images)
    min_samples = min(len(real_images), len(fake_images))
    subset_size = min(100, min_samples - 1) 
    kid = KernelInceptionDistance(feature=feature_model, subsets=10, subset_size=subset_size, normalize=True)
    kid = kid.to(DEVICE)
    kid_original = KernelInceptionDistance(subsets=10, subset_size=subset_size, normalize=True).to(DEVICE)


    real_tensors = torch.stack(real_images).to(DEVICE)
    fake_tensors = torch.stack(fake_images).to(DEVICE)

    real_loader = DataLoader(real_tensors, batch_size=16)
    fake_loader = DataLoader(fake_tensors, batch_size=16)

    # Extract features for t-SNE
    real_features = []
    fake_features = []

    with torch.no_grad():

        for batch in real_loader:
            fid.update(batch.to(DEVICE), real=True)
            fid_original.update(batch.to(DEVICE), real=True)
            kid.update(batch.to(DEVICE), real=True)
            kid_original.update(batch.to(DEVICE), real=True)
            features = feature_model(batch)
            real_features.append(features.cpu())
        for batch in fake_loader:
            fid.update(batch.to(DEVICE), real=False)
            fid_original.update(batch.to(DEVICE), real=False)
            kid.update(batch.to(DEVICE), real=False)
            kid_original.update(batch.to(DEVICE), real=False)
            features = feature_model(batch)
            fake_features.append(features.cpu())
        fid_score = fid.compute()
        fid_original_score = fid_original.compute()
        print("results for:", fake_images_path)
        print("FID:", fid_score.item())
        print("FID_original:", fid_original_score.item())
        kid_mean, kid_std = kid.compute()
        kid_original_mean, kid_original_std = kid_original.compute()
        print("KID mean:", kid_mean.item(), "±", kid_std.item())
        print("KID_original mean:", kid_original_mean.item(), "±", kid_original_std.item())

        print("Calculating extra metrics (Cosine, Euclidean, LPIPS)...")

        # Prepare features as numpy arrays if not already
        real_feats_np = real_features
        fake_feats_np = fake_features

        # LPIPS perceptual distance
        lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
        lpips_model.eval()
        num_lpips_samples = min(len(real_tensors), len(fake_tensors))
        real_lpips = real_tensors[:num_lpips_samples]
        fake_lpips = fake_tensors[:num_lpips_samples]

        lpips_scores = []
        with torch.no_grad():
            for real_img, fake_img in zip(real_lpips, fake_lpips):
                real_img = real_img.unsqueeze(0).to(DEVICE)
                fake_img = fake_img.unsqueeze(0).to(DEVICE)
                dist = lpips_model(real_img, fake_img)
                lpips_scores.append(dist.item())
        mean_lpips = np.mean(lpips_scores)
        print(f"Mean LPIPS distance: {mean_lpips:.4f}")
    # PacMAP visualization
    # Stack feature tensors
    real_features = torch.cat(real_features).numpy()
    fake_features = torch.cat(fake_features).numpy()

    # Create labels
    all_features.append(real_features)
    all_labels += ['Real'] * len(real_features)
    all_types += [cell_type] * len(real_features)
    all_features.append(fake_features)
    all_labels += ['Generated'] * len(fake_features)
    all_types += [cell_type] * len(fake_features)

# Reduce to 2D
all_features = np.vstack(all_features)
reducer = pacmap.PaCMAP(
    n_components=2,
    random_state=42,
    n_neighbors=17,
    MN_ratio=1,
    FP_ratio=2,
    )
embedding = reducer.fit_transform(all_features)

embedding = np.array(embedding)
label_array = np.array(all_labels)
type_array = np.array(all_types)

plt.figure(figsize=(9, 5), dpi=300)
for cell_type, base_color in cell_type_colors.items():
    light = base_color["real"]
    dark = base_color["fake"]

    # Real - Circles
    idx_real = (label_array == "Real") & (type_array == cell_type)
    plt.scatter(
        embedding[idx_real, 0],
        embedding[idx_real, 1],
        label=f"R - {cell_type}",
        c=[light],
        marker='o',
        edgecolors='k',
        s=80,
        linewidths=0.7,
        alpha=0.7,
        zorder=2
    )

    # Generated - Squares
    idx_fake = (label_array == "Generated") & (type_array == cell_type)
    plt.scatter(
        embedding[idx_fake, 0],
        embedding[idx_fake, 1],
        label=f"G - {cell_type}",
        c=[dark],
        marker='s',
        s=30,
        linewidths=0.3,
        alpha=0.8,
        zorder=1
    )

plt.legend(loc="best", fontsize=12, markerscale=0.9, frameon=False)
plt.xlabel("Component 1", fontsize=14,)
plt.ylabel("Component 2", fontsize=14,)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("FID/pacmap_all_types_styled.png", dpi=300)


