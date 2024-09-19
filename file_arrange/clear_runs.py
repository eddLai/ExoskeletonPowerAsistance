import os
import shutil

def check_and_clean_runs_directory(base_path):
    """
    檢查給定基礎路徑下的所有runs資料夾，確保它們包含IMU和Joint資料。
    如果不包含這些資料，則刪除該runs資料夾。

    參數:
        base_path (str): 存放runs資料夾的基礎路徑。
    """
    for run_folder in os.listdir(base_path):
        # 建立完整的路徑
        full_path = os.path.join(base_path, run_folder)
        # 確保路徑是一個資料夾
        if os.path.isdir(full_path):
            # 檢查IMU和Joint資料是否存在
            imu_exists = any('IMU' in content for content in os.listdir(full_path))
            joint_exists = any('Joint' in content for content in os.listdir(full_path))
            # 如果IMU或Joint資料不存在，刪除該runs資料夾
            if not imu_exists or not joint_exists:
                shutil.rmtree(full_path)
                print(f"Deleted {run_folder} because it lacks IMU or Joint data.")

# 使用函數的示例
runs_base_path = 'runs'  # 替換成你的runs資料夾的路徑
check_and_clean_runs_directory(runs_base_path)
