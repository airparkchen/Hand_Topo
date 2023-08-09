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
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
writer = SummaryWriter('runs/train_heat_V2_0')

# 展示模型預測結果
# def show_predictions(images, heatmaps, predictions):
#     images = images.cpu().numpy()
    
#     # Extracting the max indices for y and x coordinates separately
#     pred_joints_y = predictions.max(dim=-1)[1]
#     pred_joints_x = pred_joints_y.max(dim=-1)[1]
#     pred_joints_y = pred_joints_y.max(dim=-1)[0]

#     gt_joints_y = heatmaps.max(dim=-1)[1]
#     gt_joints_x = gt_joints_y.max(dim=-1)[1]
#     gt_joints_y = gt_joints_y.max(dim=-1)[0]

#     pred_joints_y, pred_joints_x = pred_joints_y.cpu().numpy(), pred_joints_x.cpu().numpy()
#     gt_joints_y, gt_joints_x = gt_joints_y.cpu().numpy(), gt_joints_x.cpu().numpy()

#     for i in range(images.shape[0]):
#         plt.imshow(images[i].transpose((1, 2, 0)), cmap='gray')
        
#         plt.scatter(gt_joints_x[i], gt_joints_y[i], c='r', label='Ground Truth')
#         plt.scatter(pred_joints_x[i], pred_joints_y[i], c='b', marker='x', label='Prediction')
        
#         plt.legend()
#         plt.show()

def show_predictions(img, gt_heatmap, pred_heatmap):
    """
    顯示原始圖片、ground truth 的熱圖和預測的熱圖。

    Args:
    - img: 原始圖片，Tensor of shape [batch_size, channels, height, width]
    - gt_heatmap: ground truth 的熱圖，Tensor of shape [batch_size, num_joints, height, width]
    - pred_heatmap: 預測的熱圖，同上
    """
    batch_size = img.shape[0]
    # batch_size = 1
    for idx in range(batch_size):
        img_np = img[idx].cpu().numpy().transpose(1, 2, 0)
        gt_joint = gt_heatmap[idx].sum(0).cpu().numpy()
        pred_joint = pred_heatmap[idx].sum(0).cpu().detach().numpy()
        
        # 找到ground truth和預測的關節座標
        gt_joints_pos = np.stack(np.unravel_index(gt_heatmap[idx].cpu().view(gt_heatmap[idx].shape[0], -1).argmax(1).numpy(), gt_heatmap[idx][0].shape), axis=1)
        pred_joints_pos = np.stack(np.unravel_index(pred_heatmap[idx].cpu().view(pred_heatmap[idx].shape[0], -1).argmax(1).numpy(), pred_heatmap[idx][0].shape), axis=1)
        
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].scatter(gt_joints_pos[:, 1], gt_joints_pos[:, 0], c='r', label='Ground Truth Joints')
        axes[0].scatter(pred_joints_pos[:, 1], pred_joints_pos[:, 0], c='b', label='Predicted Joints')
        axes[0].set_title('Original Image')
        axes[0].legend()

        # axes[1].imshow(gt_joint, cmap='hot', interpolation='nearest')
        # axes[1].set_title('Ground Truth Heatmap')
        axes[1].imshow(img_np)
        axes[1].scatter(gt_joints_pos[:, 1], gt_joints_pos[:, 0], c='r', label='Ground Truth Joints')
        axes[1].set_title('Predicted Heatmap')
        axes[1].legend()

        # axes[2].imshow(pred_joint, cmap='hot', interpolation='nearest')
        # axes[2].set_title('Predicted Heatmap')
        axes[2].imshow(img_np)
        axes[2].scatter(pred_joints_pos[:, 1], pred_joints_pos[:, 0], c='b', label='Predicted Joints')
        axes[2].set_title('Predicted Heatmap')
        axes[2].legend()

        axes[3].imshow(pred_joint, cmap='hot', interpolation='nearest')
        axes[3].set_title('Predicted Heatmap')
        plt.tight_layout()
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
        # 移除ResNet的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 添加多个逆卷积层
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(1024), 
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        )

        self.heatmap_layer = nn.Conv2d(64, num_joints, kernel_size=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
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
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, batch_size=8)

# 主訓練循環
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeatmapHandJointDetector().to(device)
try:
    model.load_state_dict(torch.load('heatV2_model_epoch_1.pth')) #讀取訓練檔
    print(f"using 'heatV2_model_epoch_1.pth' to train model ")
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
        # outputs = upsample(outputs)
        # print("Model outputs shape:", outputs.shape)
<<<<<<< Updated upstream
=======
        
        print("Outputs shape:", outputs.shape)
        print("Heatmaps shape:", heatmaps.shape)
        print("Loss shape:", criterion(outputs, heatmaps).shape)

        #計算關節點個別Loss
        individual_losses = criterion(outputs, heatmaps).mean(dim=(2,3))
        #OHKM
        loss = ohkm(individual_losses, top_k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
>>>>>>> Stashed changes
        # 每十批次，視覺化一次預測
        if i % 10000 == 0:
            # print(heatmaps)
            # print(outputs)
            show_predictions(images, heatmaps, outputs)
            
<<<<<<< Updated upstream

        loss = criterion(outputs, heatmaps)
        print(f"loss:{loss}")
        writer.add_scalar('Loss', loss, i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar('Loss_epoch', loss, epoch)    
=======
        print(f"loss:{loss.item()}")  # 使用.item()获得损失的具体数值
        writer.add_scalar('Loss', loss.item(), i)
    # Validation or evaluation
    model.eval()
    with torch.no_grad():
        # Your validation or evaluation code
        pass


    writer.add_scalar('Loss_epoch', loss.item(), epoch)    
>>>>>>> Stashed changes
    if pretrain == 1:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        torch.save(model.state_dict(), f'PRE_heatV2_model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), f'heatV2_model_epoch_{epoch+1}.pth')
    writer.close()
