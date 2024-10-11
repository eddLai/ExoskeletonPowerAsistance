import subprocess
import os

def run_ceinms(execution_file):
    """
    執行 CEINMS 的 Python 腳本。
    
    :param execution_file: CEINMS 的執行檔路徑 (.xml)
    """
    # 設定 CEINMS 可執行檔路徑 (需要替換成你的CEINMS可執行檔路徑)
    ceinms_executable = 'C:/Users/sean9/Desktop/gait-simulation-note/CEINMS/executable'
    
    # 檢查執行檔是否存在
    if not os.path.exists(execution_file):
        print(f"Error: {execution_file} 不存在")
        return

    # 構建命令
    command = [ceinms_executable, '--execution', execution_file]
    
    try:
        # 執行 CEINMS 模擬
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("模擬成功執行！")
        print(result.stdout.decode())
    
    except subprocess.CalledProcessError as e:
        # 如果發生錯誤，打印錯誤訊息
        print("執行時發生錯誤：")
        print(e.stderr.decode())

# 使用範例
execution_file_path = 'C:/Users/sean9/Desktop/gait-simulation-note/CEINMS/executable/executionCfg.xml'  # 替換成你的執行檔路徑
run_ceinms(execution_file_path)
