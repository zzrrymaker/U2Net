import os
import time

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from src.unet import UNet
from src import u2net_full, u2net_lite


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    weights_path = "./save_weights/unet_best_model.pth"
    img_path = "./test.png"
    output_path = "./unet_pred_result.png"
    mask_path = "./unet_pred_mask.png"

    assert os.path.exists(img_path), f"image file {img_path} dose not exists."
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.343, 0.343, 0.343), std=(0.186, 0.186, 0.186))
    ])

    origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    h, w = origin_img.shape[:2]
    img = data_transform(origin_img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

    # model = u2net_lite(out_ch=2)
    model = UNet(in_channels=3, num_classes=2, base_c=32)
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        pred = model(img)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        pred = torch.argmax(pred, dim=1)  # [1, H, W]
        pred = torch.squeeze(pred).to("cpu").numpy()  # [H, W]
        # 调整回原始图片尺寸
        pred_mask = cv2.resize(
            pred.astype(np.uint8),
            dsize=(w, h),
            interpolation=cv2.INTER_NEAREST
        )
        binary_mask = (pred_mask * 255).astype(np.uint8)
        saved_mask = cv2.imwrite(mask_path, binary_mask)
        assert saved_mask, f"failed to save prediction mask: {mask_path}"

        origin_img = np.array(origin_img, dtype=np.uint8)
        seg_img = origin_img * pred_mask[..., None]
        saved = cv2.imwrite(output_path, cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        assert saved, f"failed to save prediction: {output_path}"


if __name__ == '__main__':
    main()
