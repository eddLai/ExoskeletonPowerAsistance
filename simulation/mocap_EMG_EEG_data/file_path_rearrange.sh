# 定義基礎路徑
base_dir="/media/eddlai/DATA/ExoskeletonPowerAsistance/simulation/mocap_EMG_EEG_data"
eeg_dir="$base_dir/EMG_data_An_Yu/EEG"
emg_dir="$base_dir/EMG_data_An_Yu/EMG"
mocap_dir="$base_dir/EMG_data_An_Yu/mocap"

# 查找現有的 path 資料夾
for path_folder in "$mocap_dir"/path1_*; do
    # 確保它是一個目錄
    if [ -d "$path_folder" ]; then
        # 提取目錄名稱，例如 path1_02
        path=$(basename "$path_folder")

        # 創建對應的 EEG 和 EMG 資料夾
        mkdir -p "$mocap_dir/$path/EEG" "$mocap_dir/$path/EMG"

        # 移動對應的 EEG 文件
        eeg_file=$(ls "$eeg_dir" | grep "eeg_${path}_" | head -n 1)
        if [ -n "$eeg_file" ]; then
            mv "$eeg_dir/$eeg_file" "$mocap_dir/$path/EEG/"
            echo "Moved EEG file: $eeg_file to $path"
        else
            echo "No EEG file found for $path"
        fi

        # 移動對應的 EMG 文件
        emg_file=$(ls "$emg_dir" | grep "emg_${path}_" | head -n 1)
        if [ -n "$emg_file" ]; then
            mv "$emg_dir/$emg_file" "$mocap_dir/$path/EMG/"
            echo "Moved EMG file: $emg_file to $path"
        else
            echo "No EMG file found for $path"
        fi
    fi
done
