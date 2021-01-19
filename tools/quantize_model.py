import torch
import os
import argparse

import _init_paths
from segmentation.config import config, update_config
from segmentation.model import build_segmentation_model_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Compute size of model after quantization.')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def count_parameters(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def main():
    args = parse_args()

    # build model
    model_fp32 = build_segmentation_model_from_cfg(config)

    model_weights = torch.load(config.TEST.MODEL_FILE)

    model_fp32.load_state_dict(model_weights, strict=True)
    model_fp32.eval()

    print("Number of parameters: {}".format(count_parameters(model_fp32, trainable=False)))
    print("Number of trainable parameters: {}".format(count_parameters(model_fp32)))

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference and
    # 'qnnpack' for mobile inference. Other quantization configurations such
    # as selecting symmetric or assymetric quantization and MinMax or L2Norm
    # calibration techniques can be specified here.
    model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    # model_fp32 = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.quantization.prepare(model_fp32)
    print('Post Training Quantization: Calibration done')

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    input_fp32 = torch.randn(1, 3, 1080, 1920)
    model_fp32_prepared(input_fp32)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    print('Post Training Quantization: Convert done')

    print("Size of model after quantization")
    print_size_of_model(model_int8)

    # run the model, relevant calculations will happen in int8
    # res = model_int8(input_fp32)


if __name__ == '__main__':
    main()
