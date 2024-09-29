import os
import torch
import pickle
import argparse
import urllib.request

URL = "https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-103.pkl"

def main(args):
    """
    Merge weights of depth_sn and seg_obj models into one file.
    Args:
        args: command line arguments
    """
    # load depth_sn model weights
    print("Loading depth_sn model weights...")
    depth_sn_weights = torch.load(os.path.dirname(args.config_file) + '/model_final.pth')['model']
    # add "depth_sn_model" prefix to weights
    depth_sn_weights = {f"depth_sn_model.{k}": v for k, v in depth_sn_weights.items()}
    # load seg_obj model pretrained weights
    if args.lidar_config_file:
        print("Loading seg_obj model weights...")
        seg_obj_weights = torch.load(os.path.dirname(args.lidar_config_file) + '/model_final.pth')['model']
        seg_obj_weights = {f"seg_obj_model.{k}": v for k, v in seg_obj_weights.items()}
    else:
        print("Downloading pretrained weights...")
        with urllib.request.urlopen(URL) as f:
            seg_obj_weights = pickle.load(f)['model']
        # add "seg_obj_model.backbone.bottom_up" prefix to weights
        seg_obj_weights = {f"seg_obj_model.backbone.bottom_up.{k}": v for k, v in seg_obj_weights.items()}
    # merge weights
    print("Merging weights...")
    weights = {**depth_sn_weights, **seg_obj_weights}
    torch.save(weights, os.path.dirname(args.config_file) + '/two_stage.pth')
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", required=True)
    parser.add_argument("--lidar-config-file", default=None, metavar="FILE", help="path to lidar config file")
    args = parser.parse_args()

    main(args)
