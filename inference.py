import os
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # the Agg backend is a non-interactive backend that can only write to files.

from tqdm import tqdm

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import DatasetEvaluators
from detectron2.data import MetadataCatalog

from panoradar import get_panoradar_cfg, register_dataset, get_trainer_class, draw_vis_image, draw_range_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--mode", default="all", help="mode to visualize: all, range")
    args = parser.parse_args()

    # Config Init   
    cfg = get_panoradar_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.PIXEL_MEAN = [0 for _ in range(256)]
    cfg.MODEL.PIXEL_STD = [1 for _ in range(256)]

    # Object Detection Bounding Box Threshold for Visualization
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    # Model Init
    model = build_model(cfg)
    DetectionCheckpointer(model).load(os.path.dirname(args.config_file) + '/model_final.pth')
    print('Model loaded')

    # Dataloader Init
    register_dataset(cfg)
    trainer = get_trainer_class(cfg)(cfg)
    dataloader = trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    evaluators = DatasetEvaluators(trainer.build_evaluator(cfg, cfg.DATASETS.TEST[0]))
    print('Data loaded')

    # Output Path
    output_dir = os.path.join(os.path.dirname(args.config_file), f'vis_{args.mode}')
    os.makedirs(output_dir, exist_ok=True)

    # Inference Loop
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(dataloader):

            outputs = model(inputs)

            image_id = inputs[0]['file_name'].split('/')[-1]
            exp_id =inputs[0]['file_name'].split('/')[-3]
            building_id = inputs[0]['file_name'].split('/')[-4]
            save_filename = f'{building_id}_{exp_id}_{image_id}'.replace('.npy', '.png')

            if cfg.MODEL.META_ARCHITECTURE == 'DepthSnModel':
                draw_vis_image(inputs[0], 
                               outputs[0],
                               dataset_metadata=metadata, 
                               depth_only=True, 
                               return_rgb=False)
            elif cfg.MODEL.META_ARCHITECTURE == 'TwoStageModel':
                if args.mode == "all":
                    draw_vis_image(inputs[0], 
                                   outputs[0],
                                   dataset_metadata=metadata, 
                                   depth_only=False, 
                                   return_rgb=False)
                elif args.mode == "range":
                    draw_range_image(inputs[0],
                                     outputs[0],
                                     return_rgb=False)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path, dpi=75, transparent=False)
            plt.clf()
            plt.close()
    print('Inference Done')
