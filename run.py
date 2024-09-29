"""File to run the executables for both signal processing and machine learning parts"""

import argparse
from panoradar_SP import motion_estimation, imaging_result, process_raw_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run various scripts for PanoRadar',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("task", metavar="TASK", help="Which task to run")
    parser.add_argument("--in_building_folder", help='The input building folder. REQUIRED for motion estimation')
    parser.add_argument("--in_traj_folder", help="Desired imaging trajectory name")
    parser.add_argument("--frame_num", type=int, default=0, help="Desired imaging frame number within the trajectory")
    parser.add_argument("--out_plot_folder", default=None, help="Output path to save the imaging plot")
    parser.add_argument("--out_proc_folder", default="processed_data", help="Output folder for processed data")
    args = parser.parse_args()

    if args.task == "motion_estimation":
        motion_estimation.estimate_whole_building(args.in_building_folder)
    elif args.task == "imaging":
        imaging_result.imaging_frame_from_traj(args.in_traj_folder, args.frame_num, args.out_plot_path)
    elif args.task == "process_raw":
        process_raw_data.process_dataset(args.in_building_folder, args.out_proc_folder)
    else:
        raise ValueError("Unrecognized tasks. Should be one of {motion_estimation, imaging, process_raw}")
