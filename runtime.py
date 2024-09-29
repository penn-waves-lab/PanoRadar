"""
This script is borrowed from https://github.com/facebookresearch/detectron2/blob/main/tools/benchmark.py
and https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py with slight modifications.
"""

import tqdm
import torch
import logging
import itertools
import numpy as np

from collections import Counter

from detectron2.data import DatasetFromList
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.analysis import FlopCountAnalysis
from detectron2.utils.collect_env import collect_env_info
from detectron2.engine import default_argument_parser, launch
from detectron2.config import CfgNode, instantiate, LazyConfig

from fvcore.common.timer import Timer
from fvcore.nn import flop_count_table  # can also try flop_count_str

from panoradar import get_panoradar_cfg, register_dataset, get_trainer_class

logger = logging.getLogger("detectron2")

def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_panoradar_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.MODEL.PIXEL_MEAN = [0 for _ in range(256)]
        cfg.MODEL.PIXEL_STD = [1 for _ in range(256)]
        cfg.merge_from_list(args.opts)
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(name="fvcore")
    setup_logger()
    return cfg

def do_flop(cfg):
    trainer = get_trainer_class(cfg)
    if isinstance(cfg, CfgNode):
        data_loader = trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(100), data_loader):  # noqa
        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    logger.info(
        "Flops table computed from only one input sample:\n" + flop_count_table(flops)
    )
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}±{:.1f}".format(
            np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9
        )
    )

@torch.no_grad()
def benchmark_eval(args, cfg):
    trainer = get_trainer_class(cfg)
    if args.config_file.endswith(".yaml"):
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        data_loader = trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
    else:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

        cfg.dataloader.num_workers = 0
        data_loader = instantiate(cfg.dataloader.test)

    model.eval()
    logger.info("Model:\n{}".format(model))
    dummy_data = DatasetFromList(list(itertools.islice(data_loader, 100)), copy=False)

    def f():
        while True:
            yield from dummy_data

    for k in range(5):  # warmup
        model(dummy_data[k])

    max_iter = 300
    timer = Timer()
    with tqdm.tqdm(total=max_iter) as pbar:
        for idx, d in enumerate(f()):
            if idx == max_iter:
                break
            model(d)
            pbar.update()
    logger.info("{} iters in {} seconds.".format(max_iter, timer.seconds()))

def main() -> None:
    # global cfg, args
    parser = default_argument_parser(
        epilog="""
Examples:

$ ./runtime.py  \\
    --config-file configs/two_stage.yaml \\
    --num-gpus 1
"""
    )

    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1
    print(collect_env_info())

    # FLOPS
    cfg = setup(args)
    register_dataset(cfg)
    do_flop(cfg)

    # Inference Speed
    cfg = setup(args)
    f = benchmark_eval
    launch(
        f,
        args.num_gpus,
        args.num_machines,
        args.machine_rank,
        args.dist_url,
        args=(args, cfg),
    )

if __name__ == "__main__":
    main()  # pragma: no cover
