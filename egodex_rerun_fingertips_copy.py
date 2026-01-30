# 思路: 基于指尖位置计算末端执行器坐标系 (EEF)，并在 Rerun 中可视化。
import h5py
import numpy as np
import argparse
import os
import rerun as rr
from torchcodec.decoders import VideoDecoder

# ===========================
# 1. 骨骼定义
# ===========================
FINGER_CHAINS = {
    'Thumb': ['ThumbKnuckle', 'ThumbIntermediateBase', 'ThumbIntermediateTip', 'ThumbTip'],
    'Index': ['IndexFingerMetacarpal', 'IndexFingerKnuckle', 'IndexFingerIntermediateBase', 'IndexFingerIntermediateTip', 'IndexFingerTip'],
    'Middle': ['MiddleFingerMetacarpal', 'MiddleFingerKnuckle', 'MiddleFingerIntermediateBase', 'MiddleFingerIntermediateTip', 'MiddleFingerTip'],
    'Ring': ['RingFingerMetacarpal', 'RingFingerKnuckle', 'RingFingerIntermediateBase', 'RingFingerIntermediateTip', 'RingFingerTip'],
    'Little': ['LittleFingerMetacarpal', 'LittleFingerKnuckle', 'LittleFingerIntermediateBase', 'LittleFingerIntermediateTip', 'LittleFingerTip']
}
KNUCKLE_KEYS = ['ThumbKnuckle', 'IndexFingerKnuckle', 'MiddleFingerKnuckle', 'RingFingerKnuckle', 'LittleFingerKnuckle']
WRIST_KEY = 'Hand'

# 使用指尖定义末端平面
TIP_KEYS = {
    'Thumb': 'ThumbTip',
    'Index': 'IndexFingerTip',
    'Middle': 'MiddleFingerTip'
}

# ===========================
# 2. 核心算法: 指尖中心 EEF (左手修正版)
# ===========================
def compute_fingertip_eef(p_thumb, p_index, p_middle, side):
    """
    [修正逻辑]:
    1. 原点: 三指尖中心 (Centroid)。
    2. X轴 (红): 统一使用 -v_normal。
       - 右手: -v_normal 指向外侧 (Outward)。
       - 左手: 用户指出上一版(v_normal)反了，故改为 -v_normal。
    3. Y轴 (绿): 原点 -> 虎口中点 (保持不变)。
    4. Z轴 (蓝): X cross Y (随X自动变化)。
    """
    # 1. 计算原点 (Origin)
    centroid = (p_thumb + p_index + p_middle) / 3.0
    origin = centroid
    
    # 2. 计算 X轴 (红)
    v_thumb_index = p_index - p_thumb 
    v_thumb_middle = p_middle - p_thumb 
    
    # 原始法向量
    v_normal = np.cross(v_thumb_index, v_thumb_middle)
    
    # [修正]: 左右手统一取反
    # 这将使左手 X 轴相对于上一版发生 180 度翻转
    x_axis = -v_normal
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
    # 3. 计算 Y轴 (绿): 原点 -> (拇指+食指中点)
    midpoint_TI = (p_thumb + p_index) / 2.0
    v_y_raw = midpoint_TI - origin
    y_axis = v_y_raw / (np.linalg.norm(v_y_raw) + 1e-8)
    
    # 4. 计算 Z轴 (蓝): X cross Y
    # 由于左手 X 轴反向了，Z 轴也会随之反向
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    
    return origin, x_axis, y_axis, z_axis, midpoint_TI

def get_point(root, name, idx):
    try:
        val = root[f'transforms/{name}'][idx][:3, 3]
        if np.linalg.norm(val) < 1e-6: return None
        return val
    except KeyError:
        return None

def log_skeleton_and_eef(f, idx, side, world_to_cam):
    side_prefix = side
    
    p_thumb_tip = get_point(f, f'{side_prefix}{TIP_KEYS["Thumb"]}', idx)
    p_index_tip = get_point(f, f'{side_prefix}{TIP_KEYS["Index"]}', idx)
    p_mid_tip   = get_point(f, f'{side_prefix}{TIP_KEYS["Middle"]}', idx)
    p_wrist     = get_point(f, f'{side_prefix}{WRIST_KEY}', idx)
    
    if p_thumb_tip is None or p_index_tip is None or p_mid_tip is None:
        return None

    def to_cam(pt_world):
        return (world_to_cam @ np.append(pt_world, 1.0))[:3]

    # --- 绘制骨骼 ---
    if p_wrist is not None:
        p_w_cam = to_cam(p_wrist)
        points_cam = [p_w_cam]
        lines_cam = []
        knuckle_pts = {} 
        for finger_name, chain in FINGER_CHAINS.items():
            prev_pt_cam = p_w_cam 
            for joint_name in chain:
                full_name = f'{side_prefix}{joint_name}'
                pt_world = get_point(f, full_name, idx)
                if pt_world is not None:
                    pt_cam = to_cam(pt_world)
                    points_cam.append(pt_cam)
                    lines_cam.append([prev_pt_cam, pt_cam])
                    prev_pt_cam = pt_cam
                    if 'Knuckle' in joint_name: knuckle_pts[joint_name] = pt_cam
        sorted_knuckles = []
        for k in KNUCKLE_KEYS:
            if k in knuckle_pts: sorted_knuckles.append(knuckle_pts[k])
        if len(sorted_knuckles) > 1:
            for i in range(len(sorted_knuckles)-1):
                lines_cam.append([sorted_knuckles[i], sorted_knuckles[i+1]])
        if 'ThumbKnuckle' in knuckle_pts: lines_cam.append([p_w_cam, knuckle_pts['ThumbKnuckle']])
        
        rr.log(f"overlay/{side}/skeleton", rr.LineStrips3D(lines_cam, radii=0.002, colors=[180, 180, 180]))
        rr.log(f"overlay/{side}/joints", rr.Points3D(points_cam, radii=0.003, colors=[200, 200, 200]))

    # --- 计算 EEF ---
    origin_w, x_w, y_w, z_w, mid_ti_w = compute_fingertip_eef(p_thumb_tip, p_index_tip, p_mid_tip, side)
    
    cam_R_world = world_to_cam[:3, :3]
    origin_cam = to_cam(origin_w)
    mid_ti_cam = to_cam(mid_ti_w)
    x_cam = cam_R_world @ x_w
    y_cam = cam_R_world @ y_w
    z_cam = cam_R_world @ z_w

    # --- 绘制 ---
    p_t_cam, p_i_cam, p_m_cam = to_cam(p_thumb_tip), to_cam(p_index_tip), to_cam(p_mid_tip)
    
    # 1. 抓取三角形
    rr.log(f"overlay/{side}/grasp_triangle", rr.LineStrips3D([[p_t_cam, p_i_cam, p_m_cam, p_t_cam]], radii=0.002, colors=[255, 215, 0]))

    # 2. Y轴定义线
    rr.log(f"overlay/{side}/y_def_line", rr.LineStrips3D([[origin_cam, mid_ti_cam]], radii=0.002, colors=[255, 0, 255]))
    
    # 3. 坐标轴
    axis_len = 0.06
    rad = 0.005
    rr.log(f"overlay/{side}/X_Red", rr.Arrows3D(origins=[origin_cam], vectors=[x_cam * axis_len], colors=[255, 0, 0], radii=rad))
    rr.log(f"overlay/{side}/Y_Green", rr.Arrows3D(origins=[origin_cam], vectors=[y_cam * axis_len], colors=[0, 255, 0], radii=rad))
    rr.log(f"overlay/{side}/Z_Blue", rr.Arrows3D(origins=[origin_cam], vectors=[z_cam * axis_len], colors=[0, 0, 255], radii=rad))
    
    # 4. 原点
    rr.log(f"overlay/{side}/origin", rr.Points3D([origin_cam], radii=0.01, colors=[0, 255, 255]))

    return origin_cam

def main(args):
    rr.init("egodex_visualizer", spawn=True)
    hdf5_path = os.path.join(args.data_dir, f"{args.episode_id}.hdf5")
    mp4_path = os.path.join(args.data_dir, f"{args.episode_id}.mp4")
    
    print(f"Loading {hdf5_path}...")
    f = h5py.File(hdf5_path, 'r')
    decoder = VideoDecoder(mp4_path, device='cpu')
    num_frames = f['transforms/camera'].shape[0]
    intrinsics = f['camera/intrinsic'][:]
    H, W = decoder[0].shape[1], decoder[0].shape[2]

    rr.log("camera", rr.Pinhole(image_from_camera=intrinsics, width=W, height=H))

    print(f"Logging {num_frames} frames...")
    for t in range(num_frames):
        rr.set_time_sequence("frame_idx", t)
        rr.set_time_seconds("video_time", t / 30.0)
        
        img_np = decoder[t].numpy().transpose(1, 2, 0)
        rr.log("camera", rr.Image(img_np))
        
        cam_ext = f['transforms/camera'][t]
        world_to_cam = np.linalg.inv(cam_ext)
        
        for side in ['left', 'right']:
            log_skeleton_and_eef(f, t, side, world_to_cam)

    f.close()
    rr.save(f"egodex_{args.episode_id}.rrd")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/user/ml-egodex/test/add_remove_lid')
    parser.add_argument('--episode_id', type=str, default='0')
    parser.add_argument('--horizon', type=int, default=60)
    args = parser.parse_args()
    main(args)
    