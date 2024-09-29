import json
import numpy as np

from typing import List, Dict
from pathlib import Path
from functools import partial
from collections import OrderedDict

from detectron2.data import MetadataCatalog, DatasetCatalog

obj_name2id = OrderedDict(
    [
        ('person', 0),
        ('non-person', 1),
    ]
)

seg_classes = [
    'person',
    'chair/table',
    'railing',
    'trashcan',
    'stairs',
    'elevator',
    'door',
    'window',
    'ceiling',
    'wall',
    'floor',
]

seg_colors = [
    (0, 106, 216),
    (202, 180, 34),
    (247, 238, 177),
    (255, 138, 244),
    (213, 219, 113),
    (121, 111, 173),
    (128, 125, 186),
    (188, 189, 220),
    (102, 69, 0),
    (158, 154, 200),
    (255, 113, 151),
]

obj_colors = [[0, 106, 216], [30, 200, 31]]

metadata = {
    'stuff_classes': seg_classes,
    'stuff_colors': seg_colors,
    'ignore_label': 255,
    'thing_classes': list(obj_name2id.keys()),
}


def register_dataset(cfg):
    """Register all the custom datasets that are used.
    Leave one trajectory out:
        loto_train,  loto_test

    Leave one building out
        lobo_train_3401, lobo_train_DRL, lobo_train_annenberg, lobo_train_chem73,
        lobo_train_design, lobo_train_fisher, lobo_train_singh, lobo_train_levine,
        lobo_train_levine_north, lobo_train_moore, lobo_train_skirkanich, lobo_train_towne
        [And also change "train" to "test"]
    """
    base_path = Path(cfg.DATASETS.BASE_PATH)

    # define trajectories
    loto_train_3401 = list(base_path.glob('3401_Walnut_[ms]*/exp*-00[!0]'))
    loto_test_3401 = list(base_path.glob('3401_Walnut_[ms]*/exp*-000'))

    loto_train_DRL = list(base_path.glob('DRL_[ms]*/exp*-00[!1]'))
    loto_test_DRL = list(base_path.glob('DRL_[ms]*/exp*-001'))

    loto_train_annenberg = list(base_path.glob('annenberg_[ms]*/exp*-00[!1]'))
    loto_test_annenberg = list(base_path.glob('annenberg_[ms]*/exp*-001'))

    loto_train_chem73 = list(base_path.glob('chem73_[ms]*/exp*-00[!3]'))
    loto_test_chem73 = list(base_path.glob('chem73_[ms]*/exp*-003'))

    loto_train_design = list(base_path.glob('design_[ms]*/exp*-00[!1]'))
    loto_test_design = list(base_path.glob('design_[ms]*/exp*-001'))

    loto_train_fisher = list(base_path.glob('fisher_[ms]*/exp*-00[!1]'))
    loto_test_fisher = list(base_path.glob('fisher_[ms]*/exp*-001'))

    loto_train_singh = list(base_path.glob('singh_nanotech_[ms]*/exp*-00[!1]'))
    loto_test_singh = list(base_path.glob('singh_nanotech_[ms]*/exp*-001'))

    loto_train_levine = list(base_path.glob('levine_[sm]*/exp*-00[!1]'))
    loto_test_levine = list(base_path.glob('levine_[sm]*/exp*-001'))
    boost_levine = list(base_path.glob('levine_extra_moving/exp*'))

    loto_train_levine_north = list(base_path.glob('levine_north_[ms]*/exp*-00[!1]'))
    loto_test_levine_north = list(base_path.glob('levine_north_[ms]*/exp*-001'))
    boost_levine_north = list(base_path.glob('levine_north_extra_moving/exp*'))

    loto_train_moore = list(base_path.glob('moore_[ms]*/exp*-00[!1]'))
    loto_test_moore = list(base_path.glob('moore_[ms]*/exp*-001'))
    boost_moore = list(base_path.glob('moore_extra_moving/exp*'))

    loto_train_skirkanich = list(base_path.glob('skirkanich_[ms]*/exp*-00[!1]'))
    loto_test_skirkanich = list(base_path.glob('skirkanich_[ms]*/exp*-001'))
    boost_skirkanich = list(base_path.glob('skirkanich_extra_moving/exp*'))

    loto_train_towne = list(base_path.glob('towne_[ms]*/exp*-00[!1]'))
    loto_test_towne = list(base_path.glob('towne_[ms]*/exp*-001'))
    boost_towne = list(base_path.glob('towne_extra_moving/exp*'))

    boost_art_lib = list(base_path.glob('art_library_moving/exp*'))
    boost_houston = list(base_path.glob('houston_hall_moving/exp*'))
    boost_leidy = list(base_path.glob('leidy_lab_moving/exp*'))

    # *********************  LOTO (Leave one trajectory out)  *********************
    # fmt: off
    loto_train_all_trajs = sorted(
        loto_train_3401 + loto_train_DRL + loto_train_annenberg + loto_train_chem73 + 
        loto_train_design + loto_train_fisher + loto_train_singh + loto_train_levine + 
        loto_train_levine_north + loto_train_moore + loto_train_skirkanich + loto_train_towne +
        boost_levine + boost_levine_north + boost_moore + boost_skirkanich + boost_towne + 
        boost_art_lib + boost_houston + boost_leidy
    )
    loto_test_all_trajs = sorted(
        loto_test_3401 + loto_test_DRL + loto_test_annenberg + loto_test_chem73 +
        loto_test_design + loto_test_fisher + loto_test_singh + loto_test_levine +
        loto_test_levine_north + loto_test_moore + loto_test_skirkanich + loto_test_towne
    )
    #
    DatasetCatalog.register('loto_train', partial(get_dataset_dicts, loto_train_all_trajs))
    MetadataCatalog.get('loto_train').set(**metadata)
    DatasetCatalog.register('loto_test', partial(get_dataset_dicts, loto_test_all_trajs))
    MetadataCatalog.get('loto_test').set(
        **metadata, vis_ind=get_vis_indices(loto_test_all_trajs, loto_test_all_trajs)
    )
    # *****************************************************************************


    # **********************  LOBO (Leave one building out)  **********************
    all_trajs = loto_train_all_trajs + loto_test_all_trajs
    lobo_train_3401 = sorted([t for t in all_trajs if t not in (loto_train_3401 + loto_test_3401)])
    lobo_train_DRL = sorted([t for t in all_trajs if t not in (loto_train_DRL + loto_test_DRL)])
    lobo_train_annenberg = sorted([t for t in all_trajs if t not in (loto_train_annenberg + loto_test_annenberg)])
    lobo_train_chem73 = sorted([t for t in all_trajs if t not in (loto_train_chem73 + loto_test_chem73)])
    lobo_train_design = sorted([t for t in all_trajs if t not in (loto_train_design + loto_test_design)])
    lobo_train_fisher = sorted([t for t in all_trajs if t not in (loto_train_fisher + loto_test_fisher)])
    lobo_train_singh = sorted([t for t in all_trajs if t not in (loto_train_singh + loto_test_singh)])
    lobo_train_levine = sorted([t for t in all_trajs if t not in (loto_train_levine + loto_test_levine + boost_levine)])
    lobo_train_levine_north = sorted([t for t in all_trajs if t not in (loto_train_levine_north + loto_test_levine_north + boost_levine_north)])
    lobo_train_moore = sorted([t for t in all_trajs if t not in (loto_train_moore + loto_test_moore + boost_moore)])
    lobo_train_skirkanich = sorted([t for t in all_trajs if t not in (loto_train_skirkanich + loto_test_skirkanich + boost_skirkanich)])
    lobo_train_towne = sorted([t for t in all_trajs if t not in (loto_train_towne + loto_test_towne + boost_towne)])
    # 
    lobo_test_3401 = sorted(loto_train_3401 + loto_test_3401)
    lobo_test_DRL = sorted(loto_train_DRL + loto_test_DRL)
    lobo_test_annenberg = sorted(loto_train_annenberg + loto_test_annenberg)
    lobo_test_chem73 = sorted(loto_train_chem73 + loto_test_chem73)
    lobo_test_design = sorted(loto_train_design + loto_test_design)
    lobo_test_fisher = sorted(loto_train_fisher + loto_test_fisher)
    lobo_test_singh = sorted(loto_train_singh + loto_test_singh)
    lobo_test_levine = sorted(loto_train_levine + loto_test_levine)
    lobo_test_levine_north = sorted(loto_train_levine_north + loto_test_levine_north)
    lobo_test_moore = sorted(loto_train_moore + loto_test_moore)
    lobo_test_skirkanich = sorted(loto_train_skirkanich + loto_test_skirkanich)
    lobo_test_towne = sorted(loto_train_towne + loto_test_towne)
    #
    DatasetCatalog.register('lobo_train_3401', partial(get_dataset_dicts, lobo_train_3401))
    MetadataCatalog.get('lobo_train_3401').set(**metadata)
    DatasetCatalog.register('lobo_test_3401', partial(get_dataset_dicts, lobo_test_3401))
    MetadataCatalog.get('lobo_test_3401').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_3401, lobo_test_3401)
    )
    DatasetCatalog.register('lobo_train_DRL', partial(get_dataset_dicts, lobo_train_DRL))
    MetadataCatalog.get('lobo_train_DRL').set(**metadata)
    DatasetCatalog.register('lobo_test_DRL', partial(get_dataset_dicts, lobo_test_DRL))
    MetadataCatalog.get('lobo_test_DRL').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_DRL, lobo_test_DRL)
    )
    DatasetCatalog.register('lobo_train_annenberg', partial(get_dataset_dicts, lobo_train_annenberg))
    MetadataCatalog.get('lobo_train_annenberg').set(**metadata)
    DatasetCatalog.register('lobo_test_annenberg', partial(get_dataset_dicts, lobo_test_annenberg))
    MetadataCatalog.get('lobo_test_annenberg').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_annenberg, lobo_test_annenberg)
    )
    DatasetCatalog.register('lobo_train_chem73', partial(get_dataset_dicts, lobo_train_chem73))
    MetadataCatalog.get('lobo_train_chem73').set(**metadata)
    DatasetCatalog.register('lobo_test_chem73', partial(get_dataset_dicts, lobo_test_chem73))
    MetadataCatalog.get('lobo_test_chem73').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_chem73, lobo_test_chem73)
    )
    DatasetCatalog.register('lobo_train_design', partial(get_dataset_dicts, lobo_train_design))
    MetadataCatalog.get('lobo_train_design').set(**metadata)
    DatasetCatalog.register('lobo_test_design', partial(get_dataset_dicts, lobo_test_design))
    MetadataCatalog.get('lobo_test_design').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_design, lobo_test_design)
    )
    DatasetCatalog.register('lobo_train_fisher', partial(get_dataset_dicts, lobo_train_fisher))
    MetadataCatalog.get('lobo_train_fisher').set(**metadata)
    DatasetCatalog.register('lobo_test_fisher', partial(get_dataset_dicts, lobo_test_fisher))
    MetadataCatalog.get('lobo_test_fisher').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_fisher, lobo_test_fisher)
    )
    DatasetCatalog.register('lobo_train_singh', partial(get_dataset_dicts, lobo_train_singh))
    MetadataCatalog.get('lobo_train_singh').set(**metadata)
    DatasetCatalog.register('lobo_test_singh', partial(get_dataset_dicts, lobo_test_singh))
    MetadataCatalog.get('lobo_test_singh').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_singh, lobo_test_singh)
    )
    DatasetCatalog.register('lobo_train_levine', partial(get_dataset_dicts, lobo_train_levine))
    MetadataCatalog.get('lobo_train_levine').set(**metadata)
    DatasetCatalog.register('lobo_test_levine', partial(get_dataset_dicts, lobo_test_levine))
    MetadataCatalog.get('lobo_test_levine').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_levine, lobo_test_levine)
    )
    DatasetCatalog.register('lobo_train_levine_north', partial(get_dataset_dicts, lobo_train_levine_north))
    MetadataCatalog.get('lobo_train_levine_north').set(**metadata)
    DatasetCatalog.register('lobo_test_levine_north', partial(get_dataset_dicts, lobo_test_levine_north))
    MetadataCatalog.get('lobo_test_levine_north').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_levine_north, lobo_test_levine_north)
    )
    DatasetCatalog.register('lobo_train_moore', partial(get_dataset_dicts, lobo_train_moore))
    MetadataCatalog.get('lobo_train_moore').set(**metadata)
    DatasetCatalog.register('lobo_test_moore', partial(get_dataset_dicts, lobo_test_moore))
    MetadataCatalog.get('lobo_test_moore').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_moore, lobo_test_moore)
    )
    DatasetCatalog.register('lobo_train_skirkanich', partial(get_dataset_dicts, lobo_train_skirkanich))
    MetadataCatalog.get('lobo_train_skirkanich').set(**metadata)
    DatasetCatalog.register('lobo_test_skirkanich', partial(get_dataset_dicts, lobo_test_skirkanich))
    MetadataCatalog.get('lobo_test_skirkanich').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_skirkanich, lobo_test_skirkanich)
    )
    DatasetCatalog.register('lobo_train_towne', partial(get_dataset_dicts, lobo_train_towne))
    MetadataCatalog.get('lobo_train_towne').set(**metadata)
    DatasetCatalog.register('lobo_test_towne', partial(get_dataset_dicts, lobo_test_towne))
    MetadataCatalog.get('lobo_test_towne').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_towne, lobo_test_towne)
    )
    # fmt: on
    # *******************************************************************************


def get_dataset_dicts(traj_paths: List[Path]) -> List[Dict]:
    """Get the dataset dict from disk.

    NOTE: It only sets the file names. The dataset mapper in `mapper.py`
    will load the actual content and add them to the dict.

    Args:
        traj_paths: list of trajectory path base/building/trajectory
    Returns:
        Dataset Dict: [
           {'file_name', 'image_id', 'height', 'width',
            'depth_file_name', 'glass_file_name', 'sem_seg_file_name',
            'annotations': {'bbox', 'bbox_mode', 'segmentation', 'category_id'}
        }, ...]
    """
    dataset_dicts = []
    image_id = 0

    for traj_path in traj_paths:
        json_file_names = sorted((traj_path / Path('obj_json')).iterdir())
        rf_npy_names = sorted((traj_path / Path('rf_npy')).iterdir())
        seg_npy_names = sorted((traj_path / Path('seg_npy')).iterdir())
        lidar_npy_names = sorted((traj_path / Path('lidar_npy')).iterdir())
        glass_npy_names = sorted((traj_path / Path('glass_npy')).iterdir())

        for json_file_name, rf_npy_name, seg_npy_name, lidar_npy_name, glass_npy_name in zip(
            json_file_names, rf_npy_names, seg_npy_names, lidar_npy_names, glass_npy_names
        ):
            record = {
                'file_name': str(rf_npy_name),
                'sem_seg_file_name': str(seg_npy_name),
                'depth_file_name': str(lidar_npy_name),
                'glass_file_name': str(glass_npy_name),
                'image_id': image_id,
                'height': 64,
                'width': 512,
            }
            image_id += 1

            # read json and get object bbox
            with open(json_file_name) as f:
                items = json.load(f)
            #
            objs = []
            for item in items:
                pts = np.array(item['points'])
                px = pts[:, 0]
                py = pts[:, 1]
                label = obj_name2id[item['label']]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": 0,  # =BoxMode.XYXY_ABS
                    "segmentation": [],  # [poly],
                    "category_id": label,
                }
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def get_vis_indices(val_trajs: List[str], static_1k_trajs: List[str]) -> List[int]:
    """Get the validation indices for logging images.
    Select the first and the middle one for each trajectory.

    NOTE: Only visualize the static 1K images, otherwise there will be too many.
    This function finds the correct static 1K indices in the `val_trajs`

    Args:
        val_trajs: the validation trajectories
        static_1k_trajs: the static 1K trajectories.
    Returns:
        vis_indices: the visualization indices for logging images
    """
    num_traj_files = [len(list((traj_path / Path('rf_npy')).iterdir())) for traj_path in val_trajs]
    num_traj_files.insert(0, 0)
    traj_start_ind = np.cumsum(num_traj_files)  # [0, num_traj1, num_traj1+num_traj2, ...]

    picks = [(True if traj_path in static_1k_trajs else False) for traj_path in val_trajs]

    # select the first one and the middle one
    vis_indices = []
    for i in range(1, len(traj_start_ind)):
        if not picks[i - 1]:
            continue
        vis_indices.append(traj_start_ind[i - 1])
        vis_indices.append(traj_start_ind[i - 1] + num_traj_files[i] // 2)

    return vis_indices
