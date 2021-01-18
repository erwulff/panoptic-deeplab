# ------------------------------------------------------------------------------
# Example command:
# python tools/segment_video.py --cfg PATH_TO_CONFIG_FILE \
#   --input PATH_TO_input \
#   --output-dir PATH_TO_OUTPUT_DIR
# Originally written by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Eric Wulff
# ------------------------------------------------------------------------------


import argparse
import cv2
import os
import pprint
import logging
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.backends.cudnn as cudnn

import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
from segmentation.utils import save_annotation, save_instance_annotation, save_panoptic_annotation
import segmentation.data.transforms.transforms as T
from segmentation.data import build_test_loader_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--input',
                        help='input, could be image, image list or video',
                        required=True,
                        type=str)
    parser.add_argument('--output-dir',
                        help='output directory',
                        required=True,
                        type=str)
    parser.add_argument('--extension',
                        help='file extension if input is image list',
                        default='.png',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def read_image(file_name, format=None):
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image


class CityscapesMeta(object):
    def __init__(self):
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.label_divisor = 1000
        self.ignore_label = 255

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]  # road
        colormap[1] = [244, 35, 232]  # sidewalk
        colormap[2] = [70, 70, 70]  # building
        colormap[3] = [102, 102, 156]  # wall
        colormap[4] = [190, 153, 153]  # fence
        colormap[5] = [153, 153, 153]  # pole
        colormap[6] = [250, 170, 30]  # traffic light
        colormap[7] = [220, 220, 0]  # traffic sign
        colormap[8] = [107, 142, 35]  # vegetation
        colormap[9] = [152, 251, 152]  # terrain
        colormap[10] = [70, 130, 180]  # sky
        colormap[11] = [220, 20, 60]  # person
        colormap[12] = [255, 0, 0]  # rider
        colormap[13] = [0, 0, 142]  # car
        colormap[14] = [0, 0, 70]  # truck
        colormap[15] = [0, 60, 100]  # bus
        colormap[16] = [0, 80, 100]  # train
        colormap[17] = [0, 0, 230]  # motorcycle
        colormap[18] = [119, 11, 32]  # bicycle
        return colormap


def main():
    args = parse_args()

    logger = logging.getLogger('segment_video.py')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=args.output_dir, name='demo')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    if len(gpus) > 1:
        raise ValueError('Test only supports single core.')
    device = torch.device('cuda:{}'.format(gpus[0]))

    # build model
    model = build_segmentation_model_from_cfg(config)

    logger.info("Model:\n{}".format(model))
    model = model.to(device)
    meta_dataset = CityscapesMeta()

    # load model
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

    if os.path.isfile(model_state_file):
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
            logger.info('Evaluating a intermediate checkpoint.')
        model.load_state_dict(model_weights, strict=True)
        logger.info('Test model loaded from {}'.format(model_state_file))
    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')

    model.eval()

    # load images
    cap = None
    if os.path.exists(args.input):
        if os.path.isfile(args.input):
            # extract extension
            ext = os.path.splitext(os.path.basename(args.input))[1]
            if ext in ['.mpeg', '.mp4']:
                cap = cv2.VideoCapture(args.input)
            else:
                raise ValueError("Unsupported extension: {}.".format(ext))
        else:
            raise ValueError("Input must be a file, not a directory: {}".format(args.input))
    else:
        raise ValueError('Input file does not exists: {}'.format(args.input))

    # dir to save panoptic outputs
    panoptic_out_dir = os.path.join(args.output_dir, 'panoptic')
    PathManager.mkdirs(panoptic_out_dir)

    # build image demo transform
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                config.DATASET.MEAN,
                config.DATASET.STD
            )
        ]
    )

    # Get video information
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output_dir + '/output.avi', fourcc, fps, (width, height))

    try:
        with torch.no_grad():
            pbar = tqdm(total=length)
            ii = 0
            while(cap.isOpened()):
                ret, raw_image = cap.read()
                if ret:

                    # pad image
                    raw_shape = raw_image.shape[:2]
                    raw_h = raw_shape[0]
                    raw_w = raw_shape[1]
                    new_h = (raw_h + 31) // 32 * 32 + 1
                    new_w = (raw_w + 31) // 32 * 32 + 1
                    input_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    input_image[:, :] = config.DATASET.MEAN
                    input_image[:raw_h, :raw_w, :] = raw_image

                    image, _ = transforms(input_image, None)
                    image = image.unsqueeze(0).to(device)

                    # network
                    out_dict = model(image)
                    torch.cuda.synchronize(device)

                    # post-processing
                    semantic_pred = get_semantic_segmentation(out_dict['semantic'])

                    panoptic_pred, center_pred = get_panoptic_segmentation(
                        semantic_pred,
                        out_dict['center'],
                        out_dict['offset'],
                        thing_list=meta_dataset.thing_list,
                        label_divisor=meta_dataset.label_divisor,
                        stuff_area=config.POST_PROCESSING.STUFF_AREA,
                        void_label=(
                            meta_dataset.label_divisor *
                            meta_dataset.ignore_label),
                        threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
                        nms_kernel=config.POST_PROCESSING.NMS_KERNEL,
                        top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
                        foreground_mask=None)
                    torch.cuda.synchronize(device)

                    # Send predictions to cpu
                    semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                    panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()

                    # Crop predictions
                    semantic_pred = semantic_pred[:raw_h, :raw_w]
                    panoptic_pred = panoptic_pred[:raw_h, :raw_w]

                    # Save predictions
                    pil_image = save_panoptic_annotation(panoptic_pred, panoptic_out_dir, 'panoptic_pred_%d' % ii,
                                                         label_divisor=meta_dataset.label_divisor,
                                                         colormap=meta_dataset.create_label_colormap(),
                                                         image=raw_image)
                    ii += 1

                    # Write image to video file
                    np_image = np.asarray(pil_image)
                    np_image = np_image[:, :, ::-1]  # flip channels, OpenCV uses BGR
                    out.write(np_image)

                    # Update progress bar
                    pbar.update(1)
                else:
                    break

        pbar.close()
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception:
        logger.exception("Exception during segment_video.py:")
        raise
    finally:
        logger.info("Segmenting video finished.")
        logger.info("Panoptic predictions saved to {}".format(panoptic_out_dir))


if __name__ == '__main__':
    main()
