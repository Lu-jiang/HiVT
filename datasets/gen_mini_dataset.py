import os
import shutil

src_dir = "/mnt/workspace/gac_jianglu/HiVT/datasets/train_all/data/"
dst_dir = "/mnt/workspace/gac_jianglu/HiVT/datasets/train/data/"

# 1. 先创建目标文件夹（若不存在），避免复制时因文件夹不存在报错
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)  # 递归创建文件夹（包括父目录）
    print(f"已创建目标文件夹：{dst_dir}")

# 2. 复制前100个CSV文件（0.csv ~ 99.csv，共100个）
copied_count = 0  # 统计成功复制的文件数
for file_num in range(100):  # range(100) 生成 0~99 的整数，对应前100个文件
    src_file_path = os.path.join(src_dir, f"{file_num}.csv")
    dst_file_path = os.path.join(dst_dir, f"{file_num}.csv")

    # 检查源文件是否存在（避免因文件缺失报错）
    if os.path.isfile(src_file_path):
        # 复制文件（copy2 保留文件元信息，推荐）
        shutil.copy2(src_file_path, dst_file_path)
        print(f"已复制：{os.path.basename(src_file_path)} -> {dst_dir}")
        copied_count += 1
    else:
        print(f"警告：源文件不存在，跳过：{src_file_path}")

# 3. 打印最终结果
print(f"\n复制完成！共成功复制 {copied_count} 个文件，目标文件夹：{dst_dir}")
