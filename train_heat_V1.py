import os
import torch
import torch.nn as nn
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

# 展示模型預測結果
def show_predictions(images, heatmaps, predictions):
    images = images.cpu().numpy()
    
    # Extracting the max indices for y and x coordinates separately
    pred_joints_y = predictions.max(dim=-1)[1]
    pred_joints_x = pred_joints_y.max(dim=-1)[1]
    pred_joints_y = pred_joints_y.max(dim=-1)[0]

    gt_joints_y = heatmaps.max(dim=-1)[1]
    gt_joints_x = gt_joints_y.max(dim=-1)[1]
    gt_joints_y = gt_joints_y.max(dim=-1)[0]

    pred_joints_y, pred_joints_x = pred_joints_y.cpu().numpy(), pred_joints_x.cpu().numpy()
    gt_joints_y, gt_joints_x = gt_joints_y.cpu().numpy(), gt_joints_x.cpu().numpy()

    for i in range(images.shape[0]):
        plt.imshow(images[i].transpose((1, 2, 0)), cmap='gray')
        plt.scatter(gt_joints_x[i], gt_joints_y[i], c='r', label='Ground Truth')
        plt.scatter(pred_joints_x[i], pred_joints_y[i], c='b', marker='x', label='Prediction')
        
        plt.legend()
        plt.show()
    
# 從字符串轉換為矩陣格式
def tomatrix(s):
    if type(s) == str:
        s = s.replace("[", "").replace("]", "").replace("\n", "").split()
        s = np.array(s, dtype=float)
    return s.reshape(-1, 3)

# 使用ResNet-50建立熱圖的模型
class HeatmapHandJointDetector(nn.Module):
    def __init__(self, num_joints=42):
        super(HeatmapHandJointDetector, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # 將ResNet的全連接層移除
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 添加一個逆卷積層
        self.upsample = nn.ConvTranspose2d(2048, 2048, kernel_size=10, stride=1, padding=0)
        
        self.heatmap_layer = nn.Conv2d(2048, num_joints, kernel_size=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.upsample(x)  # 上采樣
        heatmaps = self.heatmap_layer(x)
        return heatmaps

# 根據關節的位置生成熱圖
def generate_heatmaps(joints, image_size=(320, 320), sigma=2, num_joints=42):
    heatmaps = np.zeros((num_joints, image_size[1], image_size[0]), dtype=np.float32)
    for i in range(num_joints):
        joint = joints[i]
        x = int(joint[0])
        y = int(joint[1])
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            heatmaps[i, y, x] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], ksize=(3, 3), sigmaX=sigma, sigmaY=sigma)
    return torch.tensor(heatmaps)

# 從RHD數據集讀取數據的類
class RHDDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        with open(annotations_file, 'rb') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if idx < len(self.annotations):
            img_path = os.path.join(self.img_dir, "%.5d.png" % idx)
            image = Image.open(img_path).convert('RGB')
            anno = self.annotations[f"{idx}"]
            anno["uv_vis"] = tomatrix(anno["uv_vis"])
            joints = anno["uv_vis"][:, :2]
            original_img_size = np.array([image.width, image.height])
            new_img_size = np.array([320, 320])
            scale = new_img_size / original_img_size
            joints = joints * scale
            joints = torch.tensor(joints)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                joints = self.target_transform(joints)
            heatmaps = generate_heatmaps(joints)
        else:
            raise IndexError("索引超出範圍")
        return image, heatmaps

    
from torch.utils.data import DataLoader

# 創建DataLoader的函數
def create_data_loaders(train_dataset, val_dataset, batch_size=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# 定義圖像轉換操作
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定義訓練和驗證資料集
train_dataset = RHDDataset(img_dir="../data/RHD_published_v2/training/color",
                           annotations_file="../data/RHD_published_v2/training/anno_training.json",
                           transform=transform)
val_dataset = RHDDataset(img_dir="../data/RHD_published_v2/evaluation/color",
                         annotations_file="../data/RHD_published_v2/evaluation/anno_evaluation.json",
                         transform=transform)

# 創建DataLoader
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, batch_size=4)

# 主訓練循環
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HeatmapHandJointDetector().to(device)
try:
    model.load_state_dict(torch.load('model_epoch_1.pth')) #讀取訓練檔
    print(f"using 'model_epoch_1.pth' to train model ")
    pretrain = 1
except:
    print("not using pretrain weight to train")
    pretrain = 0

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
import torch.nn as nn

upsample = nn.Upsample(size=(320, 320), mode='bilinear', align_corners=True)


num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, heatmaps) in enumerate(train_loader):
        print(f"step:{i}")
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        # print("Heatmaps shape:", heatmaps.shape)
        outputs = model(images)
        outputs = upsample(outputs)
        # print("Model outputs shape:", outputs.shape)
        # 每十批次，視覺化一次預測
        if i % 1 == 0:
            show_predictions(images, heatmaps, outputs)

        loss = criterion(outputs, heatmaps)
        print(f"loss:{loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if pretrain == 1:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        torch.save(model.state_dict(), f'PRE_heat_model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), f'heat_model_epoch_{epoch+1}.pth')
