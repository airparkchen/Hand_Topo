# compute Euclidean distance
def compute_distance(predicted_joints, ground_truth_joints):
    return ((predicted_joints - ground_truth_joints) ** 2).sum(dim=2).sqrt()  

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

def heatmap_to_coord(heatmap):
    # 找到每个heatmap的最大值的位置
    maxval_positions = torch.argmax(heatmap.view(heatmap.shape[0], heatmap.shape[1], -1), dim=2)
    # 将一维的位置转换为二维的位置
    coords = torch.stack((maxval_positions % heatmap.shape[3], maxval_positions // heatmap.shape[3]), dim=2)
    return coords


def ohkm(loss, top_k):
    """Online Hard Keypoint Mining.
    Args:
    - loss (torch.Tensor): tensor of shape (batch_size, num_joints) that represents the loss of each keypoint
    - top_k (int): number of keypoints to consider
    Returns:
    - torch.Tensor: tensor of shape (batch_size,) containing the mean loss of the hardest keypoints
    """
    ohkm_loss = 0.
    for i in range(loss.size()[0]):
        sub_loss = loss[i]
        _, topk_id = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
        tmp_loss = torch.gather(sub_loss, 0, topk_id)
        ohkm_loss += torch.mean(tmp_loss)
    ohkm_loss /= loss.size()[0]
    return ohkm_loss
