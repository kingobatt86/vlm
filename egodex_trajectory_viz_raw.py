# 机械臂末端轨迹可视化脚本（原始数据版本，无平滑处理）
# 用于生成真实的机器人训练数据
import h5py
import numpy as np
import argparse
import os
import rerun as rr
from torchcodec.decoders import VideoDecoder

# ===========================
# 1. 骨骼定义（与原脚本一致）
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

TIP_KEYS = {
    'Thumb': 'ThumbTip',
    'Index': 'IndexFingerTip',
    'Middle': 'MiddleFingerTip'
}

# ===========================
# 2. 核心算法: 指尖中心 EEF
# ===========================
def compute_fingertip_eef(p_thumb, p_index, p_middle, side):
    """
    计算末端执行器坐标系（EEF），将人手映射到机器人夹爪。
    返回: origin, x_axis, y_axis, z_axis, midpoint_TI, centroid
    """
    centroid = (p_thumb + p_index + p_middle) / 3.0
    
    v_thumb_index = p_index - p_thumb 
    v_thumb_middle = p_middle - p_thumb 
    v_normal = np.cross(v_thumb_index, v_thumb_middle)
    x_axis = -v_normal
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
    midpoint_TI = (p_thumb + p_index) / 2.0
    v_y_raw = midpoint_TI - centroid
    y_axis = v_y_raw / (np.linalg.norm(v_y_raw) + 1e-8)
    
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    
    GRIPPER_OFFSET = 0.05  # 5cm
    origin = centroid - GRIPPER_OFFSET * x_axis
    
    return origin, x_axis, y_axis, z_axis, midpoint_TI, centroid


def get_point(root, name, idx):
    """从HDF5中获取关节点位置"""
    try:
        val = root[f'transforms/{name}'][idx][:3, 3]
        if np.linalg.norm(val) < 1e-6: 
            return None
        return val
    except KeyError:
        return None


# ===========================
# 3. 提取所有帧的EEF数据（原始数据，无平滑）
# ===========================
def extract_eef_trajectory(f, num_frames, side):
    """
    从HDF5文件中提取整个轨迹的EEF数据（原始数据）
    返回: origins, rotations, valid_mask
    """
    origins = []
    rotations = []
    valid_mask = []
    
    for t in range(num_frames):
        p_thumb = get_point(f, f'{side}{TIP_KEYS["Thumb"]}', t)
        p_index = get_point(f, f'{side}{TIP_KEYS["Index"]}', t)
        p_middle = get_point(f, f'{side}{TIP_KEYS["Middle"]}', t)
        
        if p_thumb is None or p_index is None or p_middle is None:
            origins.append(np.zeros(3))
            rotations.append(np.eye(3))
            valid_mask.append(False)
        else:
            origin, x_axis, y_axis, z_axis, _, _ = compute_fingertip_eef(p_thumb, p_index, p_middle, side)
            origins.append(origin)
            # 旋转矩阵: 列向量为xyz轴
            R = np.column_stack([x_axis, y_axis, z_axis])
            rotations.append(R)
            valid_mask.append(True)
    
    return np.array(origins), rotations, np.array(valid_mask)


def interpolate_invalid_frames(origins, rotations, valid_mask):
    """
    对无效帧进行插值填充（保持最小修改）
    """
    n = len(origins)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return origins, rotations
    
    # 位置插值
    for i in range(3):
        origins[:, i] = np.interp(
            np.arange(n),
            valid_indices,
            origins[valid_indices, i]
        )
    
    # 旋转插值（简单使用最近有效帧）
    last_valid_rot = rotations[valid_indices[0]]
    for i in range(n):
        if valid_mask[i]:
            last_valid_rot = rotations[i]
        else:
            rotations[i] = last_valid_rot
    
    return origins, rotations


# ===========================
# 4. 可视化函数
# ===========================
def visualize_trajectory(origins_cam, rotations_cam, side, color_base):
    """
    可视化完整轨迹
    origins_cam: (N, 3) 相机坐标系下的位置
    rotations_cam: list of (3, 3) 相机坐标系下的旋转矩阵
    """
    n = len(origins_cam)
    
    # 1. 绘制位置轨迹曲线（一条连续的线）
    rr.log(f"trajectory/{side}/path", 
           rr.LineStrips3D([origins_cam], 
                          radii=0.003, 
                          colors=[color_base]))
    
    # 2. 绘制轨迹点（用颜色渐变表示时间）
    colors = []
    for i in range(n):
        t_ratio = i / max(n - 1, 1)
        # 从浅色到深色
        c = [int(color_base[0] * (0.3 + 0.7 * t_ratio)),
             int(color_base[1] * (0.3 + 0.7 * t_ratio)),
             int(color_base[2] * (0.3 + 0.7 * t_ratio))]
        colors.append(c)
    
    rr.log(f"trajectory/{side}/points",
           rr.Points3D(origins_cam, radii=0.005, colors=colors))
    
    # 3. 绘制起点和终点
    rr.log(f"trajectory/{side}/start",
           rr.Points3D([origins_cam[0]], radii=0.015, colors=[[0, 255, 0]]))  # 绿色起点
    rr.log(f"trajectory/{side}/end",
           rr.Points3D([origins_cam[-1]], radii=0.015, colors=[[255, 0, 0]]))  # 红色终点


def visualize_axis_trajectories(origins_cam, rotations_cam, side, axis_len=0.03, sample_interval=5):
    """
    可视化XYZ轴方向随时间的变化
    每隔sample_interval帧绘制一次坐标轴
    """
    n = len(origins_cam)
    
    x_arrows_origins, x_arrows_vectors = [], []
    y_arrows_origins, y_arrows_vectors = [], []
    z_arrows_origins, z_arrows_vectors = [], []
    
    for i in range(0, n, sample_interval):
        origin = origins_cam[i]
        R = rotations_cam[i]
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]
        
        x_arrows_origins.append(origin)
        x_arrows_vectors.append(x_axis * axis_len)
        
        y_arrows_origins.append(origin)
        y_arrows_vectors.append(y_axis * axis_len)
        
        z_arrows_origins.append(origin)
        z_arrows_vectors.append(z_axis * axis_len)
    
    # 绘制坐标轴箭头
    rr.log(f"trajectory/{side}/axes/X_Red",
           rr.Arrows3D(origins=x_arrows_origins, vectors=x_arrows_vectors, 
                      colors=[[255, 100, 100]], radii=0.002))
    rr.log(f"trajectory/{side}/axes/Y_Green",
           rr.Arrows3D(origins=y_arrows_origins, vectors=y_arrows_vectors,
                      colors=[[100, 255, 100]], radii=0.002))
    rr.log(f"trajectory/{side}/axes/Z_Blue",
           rr.Arrows3D(origins=z_arrows_origins, vectors=z_arrows_vectors,
                      colors=[[100, 100, 255]], radii=0.002))


def log_frame_with_eef(f, decoder, t, world_to_cam, origins_cam, rotations_cam, side):
    """
    逐帧可视化：视频 + 当前帧的EEF坐标轴
    """
    side_prefix = side
    
    # 获取原始指尖位置用于绘制
    p_thumb = get_point(f, f'{side_prefix}{TIP_KEYS["Thumb"]}', t)
    p_index = get_point(f, f'{side_prefix}{TIP_KEYS["Index"]}', t)
    p_middle = get_point(f, f'{side_prefix}{TIP_KEYS["Middle"]}', t)
    
    if p_thumb is None or p_index is None or p_middle is None:
        return
    
    def to_cam(pt_world):
        return (world_to_cam @ np.append(pt_world, 1.0))[:3]
    
    p_t_cam = to_cam(p_thumb)
    p_i_cam = to_cam(p_index)
    p_m_cam = to_cam(p_middle)
    
    # 抓取三角形
    rr.log(f"overlay/{side}/grasp_triangle",
           rr.LineStrips3D([[p_t_cam, p_i_cam, p_m_cam, p_t_cam]], 
                          radii=0.002, colors=[255, 215, 0]))
    
    # 当前帧EEF坐标轴
    origin_cam = origins_cam[t]
    R_cam = rotations_cam[t]
    x_cam = R_cam[:, 0]
    y_cam = R_cam[:, 1]
    z_cam = R_cam[:, 2]
    
    axis_len = 0.06
    rad = 0.005
    rr.log(f"overlay/{side}/X_Red", 
           rr.Arrows3D(origins=[origin_cam], vectors=[x_cam * axis_len], 
                      colors=[255, 0, 0], radii=rad))
    rr.log(f"overlay/{side}/Y_Green",
           rr.Arrows3D(origins=[origin_cam], vectors=[y_cam * axis_len],
                      colors=[0, 255, 0], radii=rad))
    rr.log(f"overlay/{side}/Z_Blue",
           rr.Arrows3D(origins=[origin_cam], vectors=[z_cam * axis_len],
                      colors=[0, 0, 255], radii=rad))
    
    # EEF原点
    rr.log(f"overlay/{side}/eef_origin",
           rr.Points3D([origin_cam], radii=0.01, colors=[0, 255, 255]))


# ===========================
# 5. 主函数
# ===========================
def main(args):
    rr.init("egodex_trajectory_visualizer_raw", spawn=True)
    
    hdf5_path = os.path.join(args.data_dir, f"{args.episode_id}.hdf5")
    mp4_path = os.path.join(args.data_dir, f"{args.episode_id}.mp4")
    
    print(f"Loading {hdf5_path}...")
    f = h5py.File(hdf5_path, 'r')
    decoder = VideoDecoder(mp4_path, device='cpu')
    num_frames = f['transforms/camera'].shape[0]
    intrinsics = f['camera/intrinsic'][:]
    H, W = decoder[0].shape[1], decoder[0].shape[2]
    
    rr.log("camera", rr.Pinhole(image_from_camera=intrinsics, width=W, height=H))
    
    # 颜色定义
    SIDE_COLORS = {
        'left': [100, 200, 255],   # 蓝色系
        'right': [255, 150, 100]   # 橙色系
    }
    
    # 为每只手提取原始轨迹（无平滑）
    trajectories = {}
    
    for side in ['left', 'right']:
        print(f"Extracting {side} hand trajectory (raw data)...")
        
        # 提取世界坐标系下的轨迹
        origins_world, rotations_world, valid_mask = extract_eef_trajectory(f, num_frames, side)
        
        if not np.any(valid_mask):
            print(f"  No valid frames for {side} hand, skipping...")
            continue
        
        # 仅插值填充无效帧（不做平滑处理）
        origins_world, rotations_world = interpolate_invalid_frames(
            origins_world, rotations_world, valid_mask)
        
        print(f"  Raw trajectory extracted (no smoothing applied)")
        
        # 转换到相机坐标系
        origins_cam_list = []
        rotations_cam_list = []
        
        for t in range(num_frames):
            cam_ext = f['transforms/camera'][t]
            world_to_cam = np.linalg.inv(cam_ext)
            
            # 位置转换
            origin_cam = (world_to_cam @ np.append(origins_world[t], 1.0))[:3]
            origins_cam_list.append(origin_cam)
            
            # 旋转转换
            R_cam = world_to_cam[:3, :3] @ rotations_world[t]
            rotations_cam_list.append(R_cam)
        
        origins_cam = np.array(origins_cam_list)
        
        trajectories[side] = {
            'origins_cam': origins_cam,
            'rotations_cam': rotations_cam_list,
            'origins_world': origins_world,
            'rotations_world': rotations_world,
            'valid_mask': valid_mask
        }
        
        # 可视化完整轨迹（静态）
        print(f"  Visualizing {side} trajectory...")
        visualize_trajectory(origins_cam, rotations_cam_list, side, SIDE_COLORS[side])
        visualize_axis_trajectories(origins_cam, rotations_cam_list, side, 
                                   axis_len=0.03, sample_interval=args.axis_interval)
    
    # 逐帧可视化
    print(f"Logging {num_frames} frames with video...")
    for t in range(num_frames):
        rr.set_time_sequence("frame_idx", t)
        rr.set_time_seconds("video_time", t / 30.0)
        
        # 视频帧
        img_np = decoder[t].numpy().transpose(1, 2, 0)
        rr.log("camera", rr.Image(img_np))
        
        cam_ext = f['transforms/camera'][t]
        world_to_cam = np.linalg.inv(cam_ext)
        
        # 每只手的当前帧EEF
        for side, traj in trajectories.items():
            log_frame_with_eef(f, decoder, t, world_to_cam,
                             traj['origins_cam'], traj['rotations_cam'], side)
            
            # 标记当前位置在轨迹上
            rr.log(f"trajectory/{side}/current",
                   rr.Points3D([traj['origins_cam'][t]], radii=0.012, colors=[[255, 255, 0]]))
    
    f.close()
    
    output_file = f"egodex_trajectory_raw_{args.episode_id}.rrd"
    rr.save(output_file)
    print(f"Done! Saved to {output_file}")
    print(f"NOTE: This file contains RAW trajectory data without smoothing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化机械臂末端轨迹（原始数据版本）")
    parser.add_argument('--data_dir', type=str, 
                       default='/home/user/ml-egodex/test/add_remove_lid',
                       help='数据目录路径')
    parser.add_argument('--episode_id', type=str, default='0',
                       help='Episode ID')
    parser.add_argument('--axis_interval', type=int, default=10,
                       help='坐标轴采样间隔（每隔多少帧绘制一次坐标轴）')
    args = parser.parse_args()
    main(args)
