import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import json
import csv

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

torch.cuda.empty_cache()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + ".png"
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + ".png"
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg(
        "-o",
        "--output_path",
        type=Path,
        help="Path where to save resulting masks.",
        required=True,
    )
    arg(
        "-t",
        "--tta",
        help="Test time augmentation.",
        default=None,
        choices=[None, "d4", "lr"],
    )
    arg("--rgb", help="whether output rgb images", action="store_true")
    return parser.parse_args()


def sliding_window_inference(model, image, num_classes, window_size=1024, stride=128):
    _, _, H, W = image.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_H, padded_W = image.shape

    # Initialize prediction tensor with the correct number of classes
    prediction = torch.zeros(
        (image.shape[0], num_classes, padded_H, padded_W), device=image.device
    )
    count = torch.zeros((1, 1, padded_H, padded_W), device=image.device)

    for h in range(0, padded_H - window_size + 1, stride):
        for w in range(0, padded_W - window_size + 1, stride):
            window = image[:, :, h : h + window_size, w : w + window_size]
            with torch.no_grad():
                output = model(window)
            prediction[:, :, h : h + window_size, w : w + window_size] += output
            count[:, :, h : h + window_size, w : w + window_size] += 1

    prediction /= count
    prediction = prediction[:, :, :H, :W]  # Remove padding
    return prediction


def create_region_evaluators(num_classes):
    return {
        "BB": Evaluator(num_class=num_classes),
        "NRW_1": Evaluator(num_class=num_classes),
        "NRW_3": Evaluator(num_class=num_classes),
        "SH": Evaluator(num_class=num_classes),
        "All": Evaluator(num_class=num_classes),
    }


def get_region_from_id(image_id):
    if image_id.startswith("33_"):
        return "BB"
    elif image_id.startswith("32_4"):
        return "NRW_1"
    elif image_id.startswith("32_3"):
        return "NRW_3"
    elif image_id.startswith("32_5"):
        return "SH"
    return None


def calculate_and_save_metrics(evaluators, config, output_path):
    overall_confusion_matrix = None

    for region, evaluator in evaluators.items():
        metrics = OrderedDict()
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        precision_per_class = evaluator.Precision()
        recall_per_class = evaluator.Recall()
        dice_per_class = evaluator.Dice()
        OA = evaluator.OA()

        # Get confusion matrix
        confusion_matrix = evaluator.confusion_matrix

        # Add to overall confusion matrix
        if overall_confusion_matrix is None:
            overall_confusion_matrix = confusion_matrix
        else:
            overall_confusion_matrix += confusion_matrix

        # Calculate normalized error matrix
        normalized_error_matrix = confusion_matrix / confusion_matrix.sum(
            axis=1, keepdims=True
        )

        # Per-class metrics
        for (
            class_name,
            class_iou,
            class_f1,
            class_precision,
            class_recall,
            class_dice,
        ) in zip(
            config.classes,
            iou_per_class,
            f1_per_class,
            precision_per_class,
            recall_per_class,
            dice_per_class,
        ):
            metrics[class_name] = {
                "IoU": float(class_iou),
                "F1": float(class_f1),
                "Precision": float(class_precision),
                "Recall": float(class_recall),
                "Dice": float(class_dice),
            }
            print(f"{region} - {class_name}: F1={class_f1:.4f}, IoU={class_iou:.4f}")

        # Mean metrics (excluding background class if it's the last one)
        metrics["Mean"] = {
            "mIoU": float(np.nanmean(iou_per_class[:-1])),
            "mF1": float(np.nanmean(f1_per_class[:-1])),
            "mPrecision": float(np.nanmean(precision_per_class[:-1])),
            "mRecall": float(np.nanmean(recall_per_class[:-1])),
            "mDice": float(np.nanmean(dice_per_class[:-1])),
        }

        # Overall Accuracy
        metrics["Overall_Accuracy"] = float(OA)

        # Print mean metrics
        print(f"{region} - Mean metrics (excluding background):")
        print(f"mIoU: {metrics['Mean']['mIoU']:.4f}")
        print(f"mF1: {metrics['Mean']['mF1']:.4f}")
        print(f"mPrecision: {metrics['Mean']['mPrecision']:.4f}")
        print(f"mRecall: {metrics['Mean']['mRecall']:.4f}")
        print(f"mDice: {metrics['Mean']['mDice']:.4f}")
        print(f"Overall Accuracy: {metrics['Overall_Accuracy']:.4f}")

        # Save metrics to JSON file
        output_file = output_path / f"metrics_{region}.json"
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics for {region} saved to {output_file}")

        # Save normalized error matrix to CSV file
        csv_output_file = output_path / f"normalized_error_matrix_{region}.csv"
        save_error_matrix_to_csv(
            normalized_error_matrix, config.classes, csv_output_file
        )
        print(f"Normalized error matrix for {region} saved to {csv_output_file}")

    # Calculate and save overall normalized error matrix
    overall_normalized_error_matrix = (
        overall_confusion_matrix / overall_confusion_matrix.sum(axis=1, keepdims=True)
    )
    overall_csv_output_file = output_path / "overall_normalized_error_matrix.csv"
    save_error_matrix_to_csv(
        overall_normalized_error_matrix, config.classes, overall_csv_output_file
    )
    print(f"Overall normalized error matrix saved to {overall_csv_output_file}")


def save_error_matrix_to_csv(error_matrix, class_names, file_path):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Convert class_names to a list if it's not already
        class_names_list = (
            list(class_names) if not isinstance(class_names, list) else class_names
        )
        # Write header
        writer.writerow([""] + class_names_list)
        # Write data
        for i, row in enumerate(error_matrix):
            writer.writerow([class_names_list[i]] + [f"{x:.4f}" for x in row])


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    model.cuda()
    model.eval()
    evaluators = create_region_evaluators(config.num_classes)
    # evaluator.reset()
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(
                    scales=[0.5, 0.75, 1.0, 1.25, 1.5],
                    interpolation="bicubic",
                    align_corners=False,
                ),
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):

            image = input["img"].cuda()
            image_ids = input["img_id"]
            masks_true = input["gt_semantic_seg"]

            # Perform sliding window inference
            raw_predictions = sliding_window_inference(model, image, config.num_classes)

            # Apply softmax to get class probabilities
            class_probabilities = nn.Softmax(dim=1)(raw_predictions)

            # Get final predictions (class with highest probability for each pixel)
            predictions = class_probabilities.argmax(dim=1)

            for i in range(predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                region = get_region_from_id(mask_name)

                if region:
                    evaluators[region].add_batch(
                        pre_image=mask, gt_image=masks_true[i].cpu().numpy()
                    )
                evaluators["All"].add_batch(
                    pre_image=mask, gt_image=masks_true[i].cpu().numpy()
                )

                results.append((mask, str(args.output_path / mask_name), args.rgb))

            # If you need to keep the probabilities for each class:
            # class_probabilities_np = class_probabilities.cpu().numpy()
            # You can then use class_probabilities_np for further analysis or storage

            # # raw_prediction NxCxHxW
            # raw_predictions = model(input["img"].cuda())

            # image_ids = input["img_id"]
            # masks_true = input["gt_semantic_seg"]

            # raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            # predictions = raw_predictions.argmax(dim=1)

            # for i in range(raw_predictions.shape[0]):
            #     mask = predictions[i].cpu().numpy()
            #     evaluator.add_batch(
            #         pre_image=mask, gt_image=masks_true[i].cpu().numpy()
            #     )
            #     mask_name = image_ids[i]
            #     results.append((mask, str(args.output_path / mask_name), args.rgb))

    # iou_per_class = evaluator.Intersection_over_Union()
    # f1_per_class = evaluator.F1()
    # precision_per_class = evaluator.Precision()
    # recall_per_class = evaluator.Recall()
    # dice_per_class = evaluator.Dice()
    # OA = evaluator.OA()
    # for class_name, class_iou, class_f1 in zip(
    #     config.classes, iou_per_class, f1_per_class
    # ):
    #     print("F1_{}:{}, IOU_{}:{}".format(class_name, class_f1, class_name, class_iou))
    # print(
    #     "F1:{}, mIOU:{}, OA:{}".format(
    #         np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA
    #     )
    # )
    calculate_and_save_metrics(evaluators, config, args.output_path)
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print("images writing spends: {} s".format(img_write_time))


if __name__ == "__main__":
    main()
