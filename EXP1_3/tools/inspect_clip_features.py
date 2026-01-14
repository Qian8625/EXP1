"""
检查 CLIP 文本特征文件的结构
"""
import numpy as np
import os
import sys

def inspect_npy_file(filepath):
    """检查并打印 npy 文件的结构信息"""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    print(f"\n{'='*60}")
    print(f"文件路径: {filepath}")
    print(f"{'='*60}")
    
    try:
        data = np.load(filepath)
        print(f"数据类型: {type(data)}")
        print(f"数组形状: {data.shape}")
        print(f"数据类型 (dtype): {data.dtype}")
        print(f"数组大小: {data.size}")
        print(f"内存占用: {data.nbytes / 1024 / 1024:.2f} MB")
        
        if len(data.shape) == 2:
            print(f"特征维度: {data.shape[1]}")
            print(f"样本数量: {data.shape[0]}")
        
        print(f"\n前 3 个样本的统计信息:")
        for i in range(min(3, data.shape[0])):
            sample = data[i]
            print(f"  样本 {i}:")
            print(f"    形状: {sample.shape}")
            print(f"    最小值: {sample.min():.6f}")
            print(f"    最大值: {sample.max():.6f}")
            print(f"    均值: {sample.mean():.6f}")
            print(f"    标准差: {sample.std():.6f}")
        
        print(f"\n数组统计信息:")
        print(f"  全局最小值: {data.min():.6f}")
        print(f"  全局最大值: {data.max():.6f}")
        print(f"  全局均值: {data.mean():.6f}")
        print(f"  全局标准差: {data.std():.6f}")
        
    except Exception as e:
        print(f"加载文件时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 默认数据路径
    default_data_root = "/root/cvpr26/data"
    
    # 从命令行参数获取数据路径，如果没有则使用默认值
    data_root = sys.argv[1] if len(sys.argv) > 1 else default_data_root
    
    # 检查的文件列表
    files_to_check = [
        os.path.join(data_root, "LLava_train_clip_text.npy"),
        os.path.join(data_root, "LLava_test_clip_text.npy"),
        os.path.join(data_root, "LLava_dev_clip_text.npy"),
    ]
    
    print("="*60)
    print("CLIP 文本特征文件结构检查")
    print("="*60)
    
    for filepath in files_to_check:
        inspect_npy_file(filepath)
    
    print(f"\n{'='*60}")
    print("检查完成")
    print(f"{'='*60}")

