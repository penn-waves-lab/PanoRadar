import os
import torch
import argparse
import numpy as np

from tqdm import tqdm

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import DatasetEvaluators

from panoradar import get_panoradar_cfg, register_dataset, get_trainer_class


def build_evaluator(args, cfg, dataset_name):
    """
    Build evaluator(s) for metrics
    Args:
        args (argparse.Namespace): the arguments passed in from command line
        cfg (CfgNode): the configuration file
        dataset_name (str): the name of the dataset
    Returns:
        list[HookBase]: a list of evaluator hooks
    """
    def _sem_seg_in_range_loading_fn(name, dtype):
        gt_depth = np.load(name.replace("seg_npy", "lidar_npy"))[0]
        mask_inr = (gt_depth > 0) & (gt_depth < 0.96)
        gt_semseg = np.load(name)[0].astype(dtype)
        gt_semseg[~mask_inr] = 255
        return gt_semseg

    def _sem_seg_loading_fn(name, dtype):
        return np.load(name)[0].astype(dtype)

    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    os.makedirs(output_folder, exist_ok=True)

    ret = []
    if 'depth' in args.metrics:
        from panoradar.evaluation import DepthEvaluator, AdvDepthEvaluator

        ret.append(DepthEvaluator(output_folder))
        ret.append(AdvDepthEvaluator(output_folder))
    if 'sn' in args.metrics:
        from panoradar.evaluation import SnEvaluator, AdvSnEvaluator

        ret.append(SnEvaluator(output_folder))
        ret.append(AdvSnEvaluator(output_folder))
    if 'seg' in args.metrics:
        from panoradar.evaluation import SemSegEvaluator, AdvSemSegEvaluator

        ret.append(
            SemSegEvaluator(
                dataset_name,
                output_dir=output_folder,
                sem_seg_loading_fn=_sem_seg_loading_fn,
            )
        )
        ret.append(
            AdvSemSegEvaluator(
                dataset_name,
                output_dir=output_folder,
                sem_seg_loading_fn=_sem_seg_loading_fn,
            )
        )
    if 'obj' in args.metrics:
        from panoradar.evaluation import CircularObjEvaluator, AdvCircularObjEvaluator

        ret.append(CircularObjEvaluator(dataset_name, output_dir=output_folder))
        ret.append(AdvCircularObjEvaluator(dataset_name, output_dir=output_folder))
    if 'loc' in args.metrics:
        from panoradar.evaluation import HumanLocalizationEvaluator, AdvHumanLocalizationEvaluator

        ret.append(HumanLocalizationEvaluator(output_dir=output_folder))
        ret.append(AdvHumanLocalizationEvaluator(output_dir=output_folder))

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", required=True)
    parser.add_argument('--metrics', nargs='+', default=['depth', 'sn'], help='list of evaluators')
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # Config Init
    cfg = get_panoradar_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.PIXEL_MEAN = [0 for _ in range(256)]
    cfg.MODEL.PIXEL_STD = [1 for _ in range(256)]

    # Model Init
    model = build_model(cfg)
    DetectionCheckpointer(model).load(os.path.dirname(args.config_file) + '/model_final.pth')
    print('Model loaded')

    # Dataloader Init
    register_dataset(cfg)
    trainer = get_trainer_class(cfg)
    dataloader = trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
    evaluators = DatasetEvaluators(build_evaluator(args, cfg, cfg.DATASETS.TEST[0]))
    print('Data loaded')

    # Eval Loop
    model.eval()
    evaluators.reset()
    with torch.no_grad():
        for inputs in tqdm(dataloader):

            outputs = model(inputs)
            evaluators.process(inputs, outputs)

    results = evaluators.evaluate()
    # write the results to a text file
    with open(os.path.dirname(args.config_file) + '/eval_results.txt', 'w') as f:
        for k, v in results.items():
            f.write(f"Evaluator {k} \n")
            for k2, v2 in v.items():
                f.write(f"{k2}: {v2} \n")
            f.write("\n")
    print(f'Evaluation results written to {os.path.dirname(args.config_file) + "/eval_results.txt"}')
