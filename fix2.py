import cv2
import numpy as np
import argparse
import os
import glob
from pathlib import Path

def load_image(image_path):
    """加载单张图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    return img

def get_simulated_params(image_shape, alpha=1.5):
    """
    生成模拟的鱼眼相机参数（用于演示和测试）
    
    参数:
        image_shape: 图像尺寸 (height, width) 元组
        alpha: 视野缩放系数，控制矫正后保留的视野范围
    
    返回:
        K: 相机内参矩阵 (3×3 numpy数组)
        D: 畸变系数向量 (4×1 numpy数组)
        K_new: 目标内参矩阵 (3×3 numpy数组)
        new_size: 矫正后图像尺寸 (width, height) 元组
    """
    h, w = image_shape[:2]
    
    # 尝试加载已保存的标定参数
    calib_file = 'calibration_result.npz'
    if os.path.exists(calib_file):
        try:
            data = np.load(calib_file)
            K = data['K']
            D = data['D']
            img_size = data['img_size']
            print(f"✅ 加载已保存的标定参数: {calib_file}")
            
            # 生成目标内参
            K_new = K.copy()
            K_new[0, 2] *= alpha
            K_new[1, 2] *= alpha
            new_size = (int(img_size[0] * alpha), int(img_size[1] * alpha))
            
            return K, D, K_new, new_size
        except Exception as e:
            print(f"⚠️ 加载标定参数失败: {e}")
    
    # 如果没有标定文件，尝试进行标定
    calib_folder = './chessboard_images'
    pattern_size = (7, 4)
    square_size = 1.0
    
    image_files = glob.glob(os.path.join(calib_folder, '*.jpg')) + \
                  glob.glob(os.path.join(calib_folder, '*.png'))
    image_files = sorted(image_files)
    
    if len(image_files) >= 5:
        print(f"📁 找到 {len(image_files)} 张标定图片，开始标定...")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 棋盘格3D点 (Z=0)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2).astype(np.float32)
        objp *= square_size
        
        object_points = []
        image_points = []
        img_size = None
        
        for fname in image_files:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if ret:
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
                object_points.append(objp)
                image_points.append(corners)
                print(f"  ✅ 检测成功: {os.path.basename(fname)}")
            else:
                print(f"  ❌ 检测失败: {os.path.basename(fname)}")
        
        print(f"📊 有效图片: {len(object_points)}/{len(image_files)}")
        
        if len(object_points) >= 5:
            try:
                ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                    object_points, image_points, img_size, None, None
                )
                
                if ret:
                    print("✅ 标定成功！使用真实参数")
                    print(f"   K = {K}")
                    print(f"   D = {D.ravel()}")
                    
                    K_new = K.copy()
                    K_new[0, 2] *= alpha
                    K_new[1, 2] *= alpha
                    new_size = (int(img_size[0] * alpha), int(img_size[1] * alpha))
                    
                    np.savez('calibration_result.npz', K=K, D=D, img_size=img_size)
                    return K, D, K_new, new_size
                    
            except Exception as e:
                print(f"标定过程出错: {e}")
    
    # ============================================================
    # 如果标定失败，使用模拟参数
    # ============================================================
    
    print("⚠️ 标定失败，使用模拟参数")
    
    fx = w * 0.4
    fy = h * 0.4
    cx = w / 2.0
    cy = h / 2.0
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    
    D = np.array([[-0.25], [0.08], [-0.02], [0.01]], dtype=np.float64)
    
    K_new = K.copy()
    K_new[0, 2] = cx * alpha
    K_new[1, 2] = cy * alpha
    
    new_w = int(w * alpha)
    new_h = int(h * alpha)
    new_size = (new_w, new_h)
    
    return K, D, K_new, new_size

def process_single_image(image_path, output_path, K, D, K_new, new_size):
    """
    处理单张图像：加载、矫正、保存
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径
        K: 相机内参矩阵
        D: 畸变系数
        K_new: 目标内参矩阵
        new_size: 输出图像尺寸
    """
    try:
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ❌ 无法加载: {os.path.basename(image_path)}")
            return False
        
        # 应用畸变矫正
        undistorted = cv2.undistort(img, K, D, None, K_new)
        
        # 保存结果
        cv2.imwrite(output_path, undistorted)
        print(f"  ✅ 已处理: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"  ❌ 处理失败 {os.path.basename(image_path)}: {e}")
        return False

def process_folder(input_folder, output_folder, alpha=1.5, supported_extensions=None):
    """
    批量处理文件夹中的所有图像
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        alpha: 视野缩放系数
        supported_extensions: 支持的图像扩展名列表
    
    返回:
        total_count: 总图像数
        success_count: 成功处理数
        fail_count: 失败数
    """
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 收集所有图像文件
    image_files = []
    for ext in supported_extensions:
        pattern = os.path.join(input_folder, f'*{ext}')
        image_files.extend(glob.glob(pattern))
        # 也支持大写扩展名
        pattern = os.path.join(input_folder, f'*{ext.upper()}')
        image_files.extend(glob.glob(pattern))
    
    # 去重并排序
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"⚠️ 在文件夹 '{input_folder}' 中未找到支持的图像文件")
        print(f"   支持的格式: {', '.join(supported_extensions)}")
        return 0, 0, 0
    
    print(f"📁 找到 {len(image_files)} 张图像")
    
    # 获取相机参数（使用第一张图像的尺寸）
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print("❌ 无法读取第一张图像，请检查文件")
        return 0, 0, 0
    
    print("正在获取相机参数...")
    K, D, K_new, new_size = get_simulated_params(first_img.shape, alpha)
    print(f"内参矩阵 K:\n{K}")
    print(f"畸变系数 D:\n{D.ravel()}")
    print(f"输出尺寸: {new_size[0]} x {new_size[1]}")
    print("=" * 60)
    
    # 处理所有图像
    success_count = 0
    fail_count = 0
    
    for img_path in image_files:
        # 生成输出路径（保持相对路径结构）
        rel_path = os.path.relpath(img_path, input_folder)
        output_path = os.path.join(output_folder, rel_path)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理图像
        if process_single_image(img_path, output_path, K, D, K_new, new_size):
            success_count += 1
        else:
            fail_count += 1
    
    return len(image_files), success_count, fail_count

def main():
    """
    主函数：批量处理文件夹中的所有图像
    """
    parser = argparse.ArgumentParser(
        description='批量鱼眼图像畸变矫正工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python batch_undistort.py --input ./fisheye_images --output ./undistorted_results
  python batch_undistort.py -i ./images -o ./results -a 2.0
  python batch_undistort.py -i ./images -o ./results --ext .jpg .png
        '''
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='输入文件夹路径（包含所有待处理图像）'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='输出文件夹路径（结果将保存到此目录）'
    )
    parser.add_argument(
        '--alpha', '-a', type=float, default=1.5,
        help='视野缩放系数，范围1.0-2.5，默认1.5。值越大视野越大，但边缘拉伸更明显'
    )
    parser.add_argument(
        '--ext', '-e', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
        help='支持的图像扩展名（默认: .jpg .jpeg .png .bmp .tiff .tif）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件夹 '{args.input}' 不存在")
        return
    
    # 检查输入文件夹是否为目录
    if not os.path.isdir(args.input):
        print(f"❌ 错误: '{args.input}' 不是一个文件夹")
        return
    
    # 开始批量处理
    print("=" * 60)
    print("🚀 批量鱼眼图像畸变矫正")
    print("=" * 60)
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    print(f"视野缩放系数: {args.alpha}")
    print(f"支持的格式: {', '.join(args.ext)}")
    print("=" * 60)
    
    total, success, fail = process_folder(
        args.input, 
        args.output, 
        args.alpha,
        args.ext
    )
    
    # 输出统计结果
    print("=" * 60)
    print("📊 处理完成统计:")
    print(f"  总图像数: {total}")
    print(f"  ✅ 成功: {success}")
    print(f"  ❌ 失败: {fail}")
    print(f"  输出目录: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()