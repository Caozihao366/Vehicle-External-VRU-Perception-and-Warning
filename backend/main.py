"""
车外环视感知VRU轨迹预测与异常行为预警系统
=========================================
入口文件。单路前视视频模式。

用法:
    python main.py                          # 使用默认视频
    python main.py --video test.mp4         # 指定视频
    python main.py --camera 0               # 使用摄像头
    python main.py --ego-speed 15           # 自车速度 15 m/s
    python main.py --no-lane                # 禁用车道线检测（加速）
    python main.py --cpu                    # 使用 CPU
"""
import argparse
import sys
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="车外环视VRU感知系统")
    parser.add_argument("--video", type=str, default=None,
                        help="视频文件路径（默认使用 input/test.mp4）")
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头 ID（如 0）")
    parser.add_argument("--ego-speed", type=float, default=0.0,
                        help="自车速度 (m/s)，默认 0（静止）")
    parser.add_argument("--ego-yaw-rate", type=float, default=0.0,
                        help="自车横摆角速度 (rad/s)")
    parser.add_argument("--no-lane", action="store_true",
                        help="禁用 YOLOPv2 车道线检测")
    parser.add_argument("--no-wrong-way", action="store_true",
                        help="禁用逆行检测")
    parser.add_argument("--no-sensor-quality", action="store_true",
                        help="禁用传感器质量检测")
    parser.add_argument("--cpu", action="store_true",
                        help="使用 CPU 推理")
    parser.add_argument("--output", type=str, default=None,
                        help="报文输出路径（JSONL 格式，如 output/messages.jsonl）")
    parser.add_argument("--no-ui", action="store_true",
                        help="不显示 UI 窗口（仅处理和输出报文）")
    parser.add_argument("--show-trajectories", action="store_true",
                        help="在BEV中显示自车CTRV和QCNet多模态预测轨迹")
    parser.add_argument("--no-qcnet", action="store_true",
                        help="关闭QCNet，TTC仅使用匀速回退轨迹")
    parser.add_argument("--qcnet-device", choices=("cuda", "cpu"), default=None,
                        help="QCNet推理设备；CUDA不可用时自动回退CPU")
    return parser.parse_args()


def main():
    args = parse_args()

    # 视频源
    if args.camera is not None:
        video_source = args.camera
    elif args.video is not None:
        video_source = args.video
    else:
        from system_config import DEFAULT_VIDEO, ROOT_DIR
        from pathlib import Path
        if Path(DEFAULT_VIDEO).is_file():
            video_source = DEFAULT_VIDEO
        else:
            # 回退到 A2D2 前视图片序列
            cam_front = ROOT_DIR / "input" / "CAM_FRONT"
            if cam_front.is_dir():
                video_source = str(cam_front)
                print(f"使用图片序列: {cam_front}")
            else:
                video_source = 0  # 回退到摄像头

    det_device = "cpu" if args.cpu else 0

    from pipeline import Pipeline
    from ui import UnifiedUI
    from message_builder import MessageLogger

    print("=" * 60)
    print("  车外环视感知VRU轨迹预测与异常行为预警系统")
    print("  单路前视模式 (BEV 退化为 2D)")
    print("=" * 60)

    pipeline = Pipeline(
        video_source=video_source,
        ego_speed_mps=args.ego_speed,
        ego_yaw_rate_rps=args.ego_yaw_rate,
        enable_sensor_quality=not args.no_sensor_quality,
        enable_wrong_way=not args.no_wrong_way,
        enable_yolopv2=not args.no_lane,
        det_device=det_device,
        enable_qcnet=not args.no_qcnet,
        qcnet_device=args.qcnet_device,
    )

    ui = None if args.no_ui else UnifiedUI(
        show_trajectories=args.show_trajectories
    )
    msg_logger = MessageLogger(output_path=args.output) if args.output else None

    print("\n开始处理，按 Q 退出...\n")
    try:
        for result in pipeline:
            if ui is not None:
                ui.show(result)
            if msg_logger is not None:
                msg_logger.log(result)

            if ui is not None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("用户退出。")
                    break
    except KeyboardInterrupt:
        print("\n中断。")
    finally:
        pipeline.release()
        if msg_logger is not None:
            msg_logger.close()
        cv2.destroyAllWindows()
        print("已退出。")


if __name__ == "__main__":
    main()
