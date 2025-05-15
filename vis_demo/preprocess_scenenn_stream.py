import os
import numpy as np
import cv2
from PIL import Image
import argparse

# 配置
image_folder = '/home/user/workspace/ESAM/data/scenenn/SceneNN/'
default_output_folder = '/home/user/workspace/ESAM/data/scenenn/SceneNN_stream/'
fx = 544.47329
fy = 544.47329
cx = 320
cy = 240
width = 640
height = 480

valid_sequence = ['015', '005', '030', '054', '322', '263', '243', '080', '089', '093', '096', '011']
train_sequence = ['005', '700', '207', '073', '337', '240', '237', '205', '263', '276', '014', '089', '021', '613', '260', '279', '528', '234', '096', '286', '041', '521', '217', '066', '036', '011', '065', '322', '607', '209', '255', '069', '265', '272', '092', '032', '025', '610', '054', '047', '225', '202', '076', '057', '527', '060', '273', '252', '080', '201', '231', '311', '270', '016', '251', '109', '078', '213', '227', '622', '030', '082', '294', '611', '522', '074', '087', '086', '061', '052', '623', '621', '084', '043', '062', '243', '246', '524', '098', '249', '038', '308', '609', '206', '223']

def convert_dataset(mode='train', output_root=default_output_folder):
    # 设置序列和输出路径
    if mode == 'train':
        sequences = train_sequence
        output_base = os.path.join(output_root, 'train')
        print('Converting train dataset')
    else:
        sequences = valid_sequence
        output_base = os.path.join(output_root, 'valid')
        print('Converting valid dataset')

    for sequence in sequences:
        # 为每个序列创建输出目录
        output_dir = os.path.join(output_base, sequence)
        for folder in ['color', 'depth', 'pose', 'label', 'ins']:
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

        # 保存内参矩阵
        intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1]])
        np.savetxt(os.path.join(output_dir, 'intrinsic.txt'), intrinsic)

        print(f'Processing sequence {sequence}')
        
        # 获取图像文件列表
        image_dir = os.path.join(image_folder, sequence, 'image')
        if not os.path.exists(image_dir):
            print(f'Image directory {image_dir} does not exist, skipping...')
            continue
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        frame_counter = 0
        
        for img_idx, image_file in enumerate(image_files):
            print(f'Processing frame {img_idx + 1}/{len(image_files)} in sequence {sequence}')
            
            # 提取帧索引
            frame_index = int(image_file.replace('image', '').replace('.png', ''))
            
            # 读取数据
            image_path = os.path.join(image_folder, sequence, 'image', image_file)
            depth_path = os.path.join(image_folder, sequence, 'depth', f'depth{frame_index:05d}.png')
            pose_path = os.path.join(image_folder, sequence, 'pose', f'{frame_index:05d}.npy')
            label_path = os.path.join(image_folder, sequence, 'label', f'{frame_index:05d}.npy')
            ins_path = os.path.join(image_folder, sequence, 'ins', f'{frame_index:05d}.npy')

            # 检查文件存在性
            if not all(os.path.exists(p) for p in [image_path, depth_path, pose_path]):
                print(f'Missing files for frame {frame_index} in sequence {sequence}, skipping...')
                continue

            # 读取图像、深度图和姿态
            rgb_img = np.asanyarray(Image.open(image_path), dtype=np.uint8)
            depth_img = np.asanyarray(Image.open(depth_path), dtype=np.uint16)
            depth_img = depth_img / 1000.0  # 转换为米
            pose = np.load(pose_path)

            # 保存到新格式
            # 保存RGB图像为JPG
            rgb_save = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, 'color', f'{frame_counter}.jpg'), rgb_save)

            # 保存深度图为PNG
            depth_save = (depth_img * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(output_dir, 'depth', f'{frame_counter}.png'), depth_save)

            # 保存姿态为TXT
            np.savetxt(os.path.join(output_dir, 'pose', f'{frame_counter}.txt'), pose)

            # 复制标签和实例文件
            if os.path.exists(label_path):
                label_data = np.load(label_path)
                np.save(os.path.join(output_dir, 'label', f'{frame_counter}.npy'), label_data)
            
            if os.path.exists(ins_path):
                ins_data = np.load(ins_path)
                np.save(os.path.join(output_dir, 'ins', f'{frame_counter}.npy'), ins_data)

            frame_counter += 1

        print(f'Sequence {sequence} converted with {frame_counter} frames.')

    print(f'All sequences processed for {mode} dataset.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SceneNN dataset to grouped format')
    parser.add_argument('mode', choices=['train', 'valid'], default='train', help='Dataset mode: train or valid')
    parser.add_argument('--output', default=default_output_folder, help='Output directory for converted dataset')
    args = parser.parse_args()
    
    convert_dataset(args.mode, args.output)