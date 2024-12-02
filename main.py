import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image as Img
from torchvision.transforms import ToPILImage

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=11):
        super().__init__()
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

def denoise_image(image):
    """
    Denoise the image to handle Gaussian, uniform, and impulse noise.
    :param np.ndarray image: Input image array.
    :return: Denoised image.
    """
    image = image.astype(np.float32) / 255.0

    median_denoised = cv2.medianBlur((image * 255).astype(np.uint8), 5)
    gaussian_denoised = cv2.GaussianBlur(median_denoised, (5, 5), 0)
    final_denoised = cv2.bilateralFilter(gaussian_denoised, d=9, sigmaColor=75, sigmaSpace=75)

    return final_denoised

def detect_and_segment(input_images):
    """
    :param np.ndarray input_images: N x 12288 array containing N 64x64x3 images reshaped into vectors
    :return: np.ndarray, np.ndarray
    """
    num_images = input_images.shape[0]


    pred_classes = np.zeros((num_images, 2), dtype=np.int32)
    pred_bboxes = np.zeros((num_images, 2, 4), dtype=np.float64)
    pred_seg = np.zeros((num_images, 4096), dtype=np.int32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    detection_model.to(device)
    detection_model.eval()

    segment_model = UNET(in_channels=3, out_channels=11).to(device)
    segment_model.load_state_dict(torch.load('segment.pth', map_location=device))
    segment_model.eval()

    image_converter = ToPILImage()

    for i, image_data in enumerate(input_images):
        reshaped_image = image_data.reshape(64, 64, 3)
        # denoised_image = denoise_image(reshaped_image)

        tensor_image = torch.as_tensor(reshaped_image, dtype=torch.uint8)
        tensor_image = tensor_image.permute(2, 0, 1)
        tensor_image = tensor_image.to(device)
        pil_image = image_converter(tensor_image)

        detection_output = detection_model(pil_image, size=256)
        detection_df = detection_output.pandas().xyxy[0]
        top_predictions = detection_df.sort_values(by='confidence', ascending=False).head(2)
        predicted_classes = top_predictions['class'].to_numpy()
        bounding_boxes = top_predictions[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()

        sort_order = np.argsort(predicted_classes)
        pred_classes[i] = predicted_classes[sort_order]
        pred_bboxes[i] = bounding_boxes[sort_order]
        normalized_image = tensor_image.float() / 255.0
        with torch.no_grad():
            segmentation_result = segment_model(normalized_image.unsqueeze(0))

        mask_prediction = torch.argmax(segmentation_result.squeeze(), dim=0).cpu().numpy()

        pred_seg[i] = mask_prediction.ravel()

    return pred_classes, pred_bboxes, pred_seg
