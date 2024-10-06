
import os
import fnmatch
from tkinter import Tk, filedialog
import numpy as np
import pandas as pd
from scipy.signal import find_peaks,argrelmin,butter, filtfilt, buttord, argrelextrema
from scipy.interpolate import interp1d ,PchipInterpolator
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import null_space
from scipy.interpolate import splrep, splev
from decimal import Decimal, ROUND_HALF_UP
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import json
Tasktype = 'Walking_startend'
Version = '0'

Patient_data_dir =r'C:\Users\Hermes\Desktop\NTKCAP\Patient_data'
body_parts = {
        'CHip': 0,
        'RHip': 3,
        'RKnee': 6,
        'Rankle': 9,
        'RBigToe': 12,
        'RSmallToe': 15,
        'RHeel': 18,
        'LHip': 21,
        'LKnee': 24,
        'Lankle': 27,
        'LBigToe': 30,
        'LSmallToe': 33,
        'LHeel': 36,
        'Neck': 39,
        'Nose': 45,
        'RShoulder': 48,
        'RElbow': 51,
        'RWrist': 54,
        'LShoulder': 57,
        'LElbow': 60,
        'LWrist': 63
    }
SR = 30
def round_half_up(n):
    return int(Decimal(n).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
def choose_file(directory):
    matching_files = []
    # Loop through each file in the first layer of the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Check if it is a file and if the file name does not start with 'w' or 'W'
        if os.path.isdir(file_path) and not (fnmatch.fnmatch(filename, 'Apose*') ):
            # Save the full file path to the list
            matching_files.append(file_path)
    
    # Display the matching files
    print('Files not starting with "w" or "W":')
    for file in matching_files:
        print(file)
    #import pdb;pdb.set_trace()
    return matching_files
def choose_file_GUI(Patient_data_dir):
# Create a popup window for directory selection
    def select_directory(initial_dir):
        root = Tk()
        root.withdraw()  # Hide the main Tkinter window
        directory = filedialog.askdirectory(initialdir=initial_dir, title='Select calculated folder')
        root.destroy()
        return directory

    # Initial directory (you can set this to any default path)


    # Prompt the user to select a directory (similar to uigetdir in MATLAB)
    directory = select_directory(Patient_data_dir)
    directory = os.path.normpath(directory)
    # Initialize an empty list to store the file paths
    matching_files = []

    # Loop through each file in the first layer of the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Check if it is a file and if the file name starts with 'w' or 'W'
        if os.path.isdir(file_path) and (fnmatch.fnmatch(filename, 'w*') or fnmatch.fnmatch(filename, 'W*')):
            # Save the full file path to the list
            matching_files.append(file_path)
    #import pdb;pdb.set_trace()
    # Display the matching files
    print('Files starting with "w" or "W":')
    for file in matching_files:
        print(file)
    #import pdb;pdb.set_trace()
    return matching_files
def trc_read(trc_dir):
    # Read the file
    with open(trc_dir, 'r') as file:
        rows = file.readlines()

    # Look for the start of each trace
    trace_starts = [i for i, row in enumerate(rows) if 'Scan' in row]

    # Detect numbers in the rows, and discard 'VOID' elements
    columns = []
    for row in rows:
        row_data = row.strip().split('\t')
        try:
            # Try converting to float, replace 'VOID' with np.nan
            row_data = [float(x) if x != 'VOID' else np.nan for x in row_data]
            columns.append(row_data)
        except ValueError:
            # Skip rows that cannot be converted to floats
            continue

    # Transpose columns for consistency with MATLAB code
    columns = [list(np.transpose(column)) for column in columns]

    # Store individual traces in a list
    traces = []
    for k in range(len(trace_starts) - 1):
        trace = np.array(columns[trace_starts[k] + 4:trace_starts[k + 1] - 1])
        traces.append(trace)

    # Extract data
    data = []
    for column in columns[0:]:
        if len(column) > 2:  # Ensure there are enough columns to slice
            data.append(column[2:])  # Skip the first two columns

    # Convert data to a numpy array
    data = np.array(data)

    return data
def motfile_read(IK_dir):
    with open(IK_dir, 'r') as file:
        lines = file.readlines()

    # Find the line number of the 'endheader'
    header_end_line = 0
    for i, line in enumerate(lines):
        if 'endheader' in line:
            header_end_line = i
            break

    # The actual data starts after the 'endheader' line
    data_start_line = header_end_line + 1

    # Read the data using pandas, skipping the header lines
    angle = pd.read_csv(IK_dir, delimiter='\t', skiprows=data_start_line)
    return angle

def COM_define(trc_dir):
    data = trc_read(trc_dir)
    
    CHip = 0
    RHip = 3
    RKnee = 6
    Rankle = 9
    RBigToe = 12
    RSmallToe = 15
    RHeel = 18
    LHip = 21
    LKnee = 24
    Lankle = 27
    LBigToe = 30
    LSmallToe = 33
    LHeel = 36
    Neck = 39
    Nose = 45
    RShoulder = 48
    RElbow = 51
    RWrist = 54
    LShoulder = 57
    LElbow = 60
    LWrist = 63
    
    Upper_arm_R = (data[:, RShoulder:RShoulder+3] + (data[:, RElbow:RElbow+3] - data[:, RShoulder:RShoulder+3]) * 0.436) * 0.028
    Upper_arm_L = (data[:, LShoulder:LShoulder+3] + (data[:, LElbow:LElbow+3] - data[:, LShoulder:LShoulder+3]) * 0.436) * 0.028
    Forearm_R = (data[:, RElbow:RElbow+3] + (data[:, RWrist:RWrist+3] - data[:, RElbow:RElbow+3]) * 0.682) * 0.022
    Forearm_L = (data[:, LElbow:LElbow+3] + (data[:, LWrist:LWrist+3] - data[:, LElbow:LElbow+3]) * 0.682) * 0.022
    Foot_R = (data[:, Rankle:Rankle+3] + ((data[:, RBigToe:RBigToe+3] + data[:, RSmallToe:RSmallToe+3]) / 2 - data[:, Rankle:Rankle+3]) * 0.5) * 0.0145
    Foot_L = (data[:, Lankle:Lankle+3] + ((data[:, LBigToe:LBigToe+3] + data[:, LSmallToe:LSmallToe+3]) / 2 - data[:, Lankle:Lankle+3]) * 0.5) * 0.0145
    Leg_R = (data[:, RKnee:RKnee+3] + (data[:, Rankle:Rankle+3] - data[:, RKnee:RKnee+3]) * 0.433) * 0.0465
    Leg_L = (data[:, LKnee:LKnee+3] + (data[:, Lankle:Lankle+3] - data[:, LKnee:LKnee+3]) * 0.433) * 0.0465
    Thigh_R = (data[:, RHip:RHip+3] + (data[:, RKnee:RKnee+3] - data[:, RHip:RHip+3]) * 0.433) * 0.1
    Thigh_L = (data[:, LHip:LHip+3] + (data[:, LKnee:LKnee+3] - data[:, LHip:LHip+3]) * 0.433) * 0.1
    Head = (data[:, Neck:Neck+3] + (data[:, Nose:Nose+3] - data[:, Neck:Neck+3]) * 1) * 0.081
    Trunk_R = (data[:, RHip:RHip+3] + (data[:, RShoulder:RShoulder+3] - data[:, RHip:RHip+3]) * 0.5) * 0.2485
    Trunk_L = (data[:, LHip:LHip+3] + (data[:, LShoulder:LShoulder+3] - data[:, LHip:LHip+3]) * 0.5) * 0.2485

    cm = Upper_arm_R + Upper_arm_L + Forearm_R + Forearm_L + Foot_R + Foot_L + Leg_R + Leg_L + Thigh_R + Thigh_L + Head + Trunk_R + Trunk_L
    
    return cm

def heel_v_xy_plane(trc_dir, SRd):
    def downsample(data, factor):
        return data[::factor]

    def islocalmin(data):
    # Use find_peaks to find local minima, and invert the data
        peaks, _ = find_peaks(-data)
        return peaks
    # Load and process data
    data = trc_read(trc_dir)
    
    Rheel = 18  # Adjust these names to match your data columns
    Lheel = 36  # Adjust these names to match your data columns

    SR = 30  # fps
    down_index = int(SR / SRd)
    
    R_heel = data[:,Rheel:Rheel+3]
    L_heel = data[:,Lheel:Lheel+3]
    
    R_heel_d = downsample(R_heel, down_index)
    L_heel_d = downsample(L_heel, down_index)
    
    # Right heel velocity calculation
    x = R_heel[:, 0]
    y = R_heel[:, 2]
    x1 = np.append(x[0], x[:-1])
    y1 = np.append(y[0], y[:-1])
    vx = x - x1
    vy = y - y1
    v = np.sqrt(vx**2 + vy**2) * SR
    TF = islocalmin(v)
    loc = TF
    minima = v[loc]
    vr = v
    
    # Left heel velocity calculation
    x = L_heel[:, 0]
    y = L_heel[:, 2]
    x1 = np.append(x[0], x[:-1])
    y1 = np.append(y[0], y[:-1])
    vx = x - x1
    vy = y - y1
    v = np.sqrt(vx**2 + vy**2) * SR
    TF = islocalmin(v)
    loc = TF
    minima = v[loc]
    vl = v

    return vr, vl
def gaitspeedcm(trc_dir, SRd):
    def downsample(data, factor):
        return data[::factor]
    # Load and process data
    data = trc_read(trc_dir)
    SR = 30  # fps
    
    # Calculate center of mass using COM_define
    cm = COM_define(trc_dir)
    down_index = int(SR / SRd)
    C_Hip = cm
    
    # Downsample the center of mass data
    C_Hip_d = downsample(C_Hip, down_index)
    
    # Calculate gait speed
    x = C_Hip_d[:, 0]
    y = C_Hip_d[:, 2]
    
    x1 = np.append(x[0], x[:-1])
    y1 = np.append(y[0], y[:-1])
    
    vx = x - x1
    vy = y - y1
    
    v = np.sqrt(vx**2 + vy**2) * SRd
    
    # Get maximum speed
    max_speed = np.max(v)
    
    return v
def create_post_analysis_dir(dir_task):
    post_analysis_dir = os.path.join(dir_task, 'post_analysis')
    
    # Check if the folder exists
    if not os.path.exists(post_analysis_dir):
        # If the folder does not exist, create it
        os.makedirs(post_analysis_dir)
        print(f'Folder created: {post_analysis_dir}')
    else:
        print(f'Folder already exists: {post_analysis_dir}')
def find_foot_strike(data,vr30,vl30,SR,dir_task,title):
    
    vR = vr30
    vL = vl30
    xR = data[:,18]
    yR =  data[:,20]
    xL = data[:,36]
    yL =  data[:,38]
    RHeel = body_parts['RHeel']
    RSmallToe =body_parts['RSmallToe']
    RBigToe =body_parts['RBigToe']
    LHeel= body_parts['LHeel']
    LSmallToe= body_parts['LSmallToe']
    LBigToe= body_parts['LBigToe']
    # Calculate tempR and tempL
    tempR = (data[:, [RHeel, RHeel + 2]] + data[:, [RSmallToe, RSmallToe + 2]] + data[:, [RBigToe, RBigToe + 2]]) / 3
    tempL = (data[:, [LHeel, LHeel + 2]] + data[:, [LSmallToe, LSmallToe + 2]] + data[:, [LBigToe, LBigToe + 2]]) / 3

    # Calculate vR and vL
    vR = np.sqrt(np.sum(np.diff(tempR, axis=0)**2, axis=1)) * SR
    vL = np.sqrt(np.sum(np.diff(tempL, axis=0)**2, axis=1)) * SR


    aim = 1
    # Analyze right heel velocity
    

    TR_R = np.mean(vR) + np.std(vR)
    locs_R,a = find_peaks(vR)
    pks_R = vR[locs_R]
    TF_R = argrelmin(vR)[0]
    locs_minR = TF_R[vR[TF_R] < np.mean(vR) - 0.2 * np.std(vR)]
    final_temp_locs_R = locs_R[pks_R > TR_R]
    final_locs_R = []

    for i in range(len(final_temp_locs_R) - 1):
        if np.min(vR[final_temp_locs_R[i]:final_temp_locs_R[i + 1]]) < np.mean(vR) - 0.2 * np.std(vR):
            final_locs_R.append(final_temp_locs_R[i])
    if len(final_temp_locs_R) > 1 and np.min(vR[final_temp_locs_R[-2]:final_temp_locs_R[-1]]) < np.mean(vR) - 0.2 * np.std(vR):
        final_locs_R.append(final_temp_locs_R[-1])
    
    locs_possible_min_R = np.where((np.abs(np.diff(vR) / np.std(np.diff(vR))) < 0.3))[0]
    locs_possible_min_R = locs_possible_min_R[vR[locs_possible_min_R] < np.mean(vR) - 0.2 * np.std(vR)]
    locs_possible_min_R = np.sort(np.concatenate((locs_minR, locs_possible_min_R)))

    p_R = []
    n_R = []

    for loc in final_locs_R:
        ptemp_R = np.where(locs_possible_min_R - loc > 0)[0]
        ntemp_R = np.where(locs_possible_min_R - loc < 0)[0]
        if ptemp_R.size > 0:
            p_R.append(ptemp_R[0])
        if ntemp_R.size > 0:
            n_R.append(ntemp_R[-1])
    

    # Save results to All array (assuming All is a global variable or passed as an argument)
    # All[aim, 37] = locs_possible_min_R[n_R]
    # All[aim, 23] = [locs_possible_min_R[n_R[0]], locs_possible_min_R[p_R]]

    # Analyze left heel velocity
    TR_L = np.mean(vL) + np.std(vL)

    locs_L,a = find_peaks(vL)
    pks_L = vL[locs_L]
    TF_L = argrelmin(vL)[0]
    locs_minL = TF_L[vL[TF_L] < np.mean(vL) - 0.2 * np.std(vL)]
    final_temp_locs_L = locs_L[pks_L > TR_L]
    final_locs_L = []

    for i in range(len(final_temp_locs_L) - 1):
        if np.min(vL[final_temp_locs_L[i]:final_temp_locs_L[i + 1]]) < np.mean(vL) - 0.2 * np.std(vL):
                final_locs_L.append(final_temp_locs_L[i])
    if len(final_temp_locs_L) > 1 and np.min(vL[final_temp_locs_L[-2]:final_temp_locs_L[-1]]) < np.mean(vL) - 0.2 * np.std(vL):
            final_locs_L.append(final_temp_locs_L[-1])
    
    locs_possible_min_L = np.where((np.abs(np.diff(vL) / np.std(np.diff(vL))) < 0.3))[0]
    locs_possible_min_L = locs_possible_min_L[vL[locs_possible_min_L] < np.mean(vL) - 0.2 * np.std(vL)]
    locs_possible_min_L = np.sort(np.concatenate((locs_minL, locs_possible_min_L)))

    p_L = []
    n_L = []

    for loc in final_locs_L:
        ptemp_L = np.where(locs_possible_min_L - loc > 0)[0]
        ntemp_L = np.where(locs_possible_min_L - loc < 0)[0]
        if ptemp_L.size > 0:
            p_L.append(ptemp_L[0])
        if ntemp_L.size > 0:
            n_L.append(ntemp_L[-1])
    
    
    # Save results to All array (assuming All is a global variable or passed as an argument)
    # All[aim, 38] = locs_possible_min_L[n_L]
    # All[aim, 24] = [locs_possible_min_L[n_L[0]], locs_possible_min_L[p_L]]

    # Final plot adjustments
    
    #plt.show()
    # Assuming R_locs_possible_min_n and L_locs_possible_min_n are numpy arrays
    R_locs_possible_min_n = locs_possible_min_R[n_R]  # replace with actual data
    L_locs_possible_min_n = locs_possible_min_L[n_L] # replace with actual data

    R_locs_possible_min_p = locs_possible_min_R[p_R]  # replace with actual data
    L_locs_possible_min_p = locs_possible_min_L[p_L]  # replace with actual data

    a = np.argmin([R_locs_possible_min_n[0], L_locs_possible_min_n[0]]) + 1
    count1 = 1
    count2 = 1

    while count1 < len(R_locs_possible_min_n)- 1 and count2 < len(L_locs_possible_min_n)-1:
        if a == 1:
            temp = np.where((L_locs_possible_min_n > R_locs_possible_min_n[count1]) & 
                            (L_locs_possible_min_n < R_locs_possible_min_n[count1 + 1]))[0]
            if len(temp) > 1:
                L_locs_possible_min_n = np.delete(L_locs_possible_min_n, temp[1:])
                L_locs_possible_min_p = np.delete(L_locs_possible_min_p, temp[:-1])
                a = 2
            elif len(temp) == 0:
                R_locs_possible_min_n = np.delete(R_locs_possible_min_n, count1 + 1)
                R_locs_possible_min_p = np.delete(R_locs_possible_min_p, count1 + 1)
                a = 1
                count1 -= 1
            else:
                a = 2
            count1 += 1
        else:
            temp = np.where((R_locs_possible_min_n > L_locs_possible_min_n[count2]) & 
                            (R_locs_possible_min_n < L_locs_possible_min_n[count2 + 1]))[0]
            if len(temp) > 1:
                R_locs_possible_min_n = np.delete(R_locs_possible_min_n, temp[1:])
                R_locs_possible_min_p = np.delete(R_locs_possible_min_p, temp[:-1])
                a = 1
            elif len(temp) == 0:
                L_locs_possible_min_n = np.delete(L_locs_possible_min_n, count2 + 1)
                L_locs_possible_min_p = np.delete(L_locs_possible_min_p, count2 + 1)
                count2 -= 1
                a = 2
            else:
                a = 1
            count2 += 1
        fig, axs = plt.subplots(2, 1, figsize=(15, 9))
    axs[0].plot(vR, linewidth=1.5)
    axs[0].scatter(R_locs_possible_min_n , vR[R_locs_possible_min_n ], color='r', s=100)
    axs[0].scatter(R_locs_possible_min_p, vR[R_locs_possible_min_p], color='y', s=100)
    axs[0].scatter(final_locs_R, vR[final_locs_R], color='g', s=100)
    #import pdb;pdb.set_trace()
    try:
        axs[0].axhline(np.mean(vR) - 0.2 * np.std(vR), color='k', linestyle='--')
    except Exception as e:
        print(f"An error occurred: {e}")
    
    axs[0].axhline(np.mean(vR) - 0.2 * np.std(vR), color='k', linestyle='--')
    axs[1].plot(vL, linewidth=1.5)
    axs[1].scatter(L_locs_possible_min_n , vL[L_locs_possible_min_n ], color='r', s=100)
    axs[1].scatter(L_locs_possible_min_p, vL[L_locs_possible_min_p], color='y', s=100)
    axs[1].scatter(final_locs_L, vL[final_locs_L], color='g', s=100)
    axs[1].axhline(np.mean(vL) - 0.2 * np.std(vL), color='k', linestyle='--')
    axs[0].set_title('Right Heel Velocity')
    axs[1].set_title('Left Heel Velocity')
    plt.suptitle('Find Heel Strike'  +title, fontsize=20)
    #plt.savefig(f'{post_analysis_dir}/Heel_segment.png')
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','Find Heel Strike'  +title+ '.png'))
    frame_R_heel_sground = np.concatenate((R_locs_possible_min_n[:1], R_locs_possible_min_p))
    frame_L_heel_sground = np.concatenate((L_locs_possible_min_n[:1], L_locs_possible_min_p))
    frame_R_heel_lground = R_locs_possible_min_n 
    frame_L_heel_lground = L_locs_possible_min_n 
    
    return frame_R_heel_sground,frame_L_heel_sground,frame_R_heel_lground,frame_L_heel_lground
def initial_read_data(IK_dir,trc_dir,dir_task,title):
    
    data = trc_read(trc_dir)
    angle = motfile_read(IK_dir)
    cm = COM_define(trc_dir)
    SRd = 30  # Sampling rate in data
    vr30, vl30 = heel_v_xy_plane(trc_dir, SRd)
    vcm30 = gaitspeedcm(trc_dir, SRd)
    frame_R_heel_sground,frame_L_heel_sground,frame_R_heel_lground,frame_L_heel_lground =find_foot_strike(data,vr30,vl30,SR,dir_task,title)
    
    return data ,angle,cm,vr30,vl30,vcm30,frame_R_heel_sground,frame_L_heel_sground,frame_R_heel_lground,frame_L_heel_lground
def COM_analysis(cm,frame_R_heel_sground,dir_task,title):






    def project_point_onto_plane(c, P, n):
        return c - np.dot(c - P, n) * n

    def roty(theta):
        theta = np.deg2rad(theta)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        return R

    # Assuming cm is a numpy array of shape (N, 3)



    def fit_poly1(x, y):
        coeffs = np.polyfit(x, y, 1)
        return np.poly1d(coeffs)

    def project_point_onto_plane(c, P, n):
        return c - np.dot(c - P, n) * n

    def roty(theta):
        theta = np.deg2rad(theta)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
        return R
    def lowpass_filter(data, cutoff_freq, fs, steepness):
        nyquist = 0.5 * fs
        fpass = cutoff_freq  # Passband frequency in Hz

        # Calculate the transition width based on steepness
        W = (1 - steepness) * (nyquist - fpass)
        fstop = fpass + W  # Stopband frequency in Hz

        # Normalize the frequencies
        wpass = fpass / nyquist
        wstop = fstop / nyquist

        # Passband ripple and stopband attenuation (in dB)
        gpass = 0.1  # Passband ripple in dB
        gstop = 60  # Stopband attenuation in dB

        # Determine the order and natural frequency
        N, Wn = buttord(wpass, wstop, gpass, gstop)

        # Design the Butterworth filter
        b, a = butter(N, Wn, btype='low', analog=False)

        # Apply the filter
        y = filtfilt(b, a, data)
        return y  # Return the filtered data and the filter order
    # Assuming cm is a numpy array of shape (N, 3)
    def islocalmin(data):
        return np.r_[True, data[1:] < data[:-1]] & np.r_[data[:-1] < data[1:], True]

    add = 20
    add_lateral_only = 30
    s25 = frame_R_heel_sground[round_half_up(len(frame_R_heel_sground) / 2)-1]
    e75 = frame_R_heel_sground[round_half_up(len(frame_R_heel_sground) / 2) + 1-1]
    s25edit = s25 - add
    e75edit = e75 + add
    s25edit_lateral = 25 - add_lateral_only
    e75edit_lateral = e75 + add_lateral_only

    # Polynomial fit
    fitobject = fit_poly1(cm[:, 0], cm[:, 2])
    a = np.array([0, fitobject(0)])
    b = np.array([1, fitobject(1)])
    line_vec = a - b
    line_vec = np.array([a[0] - b[0], a[1] - b[1]])
    v1 = null_space(line_vec.reshape(1, -1)).flatten()
    v1 = np.array([v1[0], 0, v1[1]])
    v2 = np.array([0, 1, 0])
    P = cm[-1, :]
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot([0, v1[0]], [0, v1[2]], [0, v1[1]])
    ax.plot([0, v2[0]], [0, v2[2]], [0, v2[1]])
    ax.plot(cm[[0, -1], 0], cm[[0, -1], 2], cm[[0, -1], 1])

    C_proj_all = []
    for c in cm:
        C_proj = project_point_onto_plane(c, P, n)
        C_proj_all.append(C_proj)
    C_proj_all = np.array(C_proj_all)

    plt.plot(cm[:, 0], cm[:, 2])
    plt.close()

    fig =  plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(121, projection='3d')

    direction = cm[e75edit, 0] - cm[s25edit, 0]
    if direction > 0:
        ax1.scatter(C_proj_all[s25edit:e75edit+1, 0], -C_proj_all[s25edit:e75edit+1, 2], C_proj_all[s25edit:e75edit+1, 1])
        ax1.plot(cm[s25edit:e75edit+1, 0], -cm[s25edit:e75edit+1, 2], cm[s25edit:e75edit+1, 1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
    else:
        ax1.scatter(C_proj_all[s25edit:e75edit+1, 0], C_proj_all[s25edit:e75edit+1, 2], C_proj_all[s25edit:e75edit+1, 1])
        ax1.plot(cm[s25edit:e75edit+1, 0], cm[s25edit:e75edit+1, 2], cm[s25edit:e75edit+1, 1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

    A = abs(C_proj_all[0, 0] - C_proj_all[70, 0])
    B = abs(C_proj_all[0, 2] - C_proj_all[70, 2])
    D = np.arctan2(A, B)
    D = np.rad2deg(D)
    print(D)

    R = roty(D)
    print(R)

    
    rotate_C_proj_all = C_proj_all @ R

    # Plotting rotated data
    direction = cm[e75edit, 0] - cm[s25edit, 0]
    ax2 = fig.add_subplot(122)
    
    if direction > 0:
        ax2.plot(rotate_C_proj_all[s25edit:e75edit+1, 2], rotate_C_proj_all[s25edit:e75edit+1, 1], linewidth=1.5)
        center_plotz = np.mean(rotate_C_proj_all[s25edit:e75edit+1, 1])
        center_ploty = np.mean(rotate_C_proj_all[s25edit:e75edit+1, 2])
    else:
        ax2.plot(-rotate_C_proj_all[s25edit:e75edit+1, 2], rotate_C_proj_all[s25edit:e75edit+1, 1], linewidth=1.5)
        center_plotz = np.mean(rotate_C_proj_all[s25edit:e75edit+1, 1])
        center_ploty = np.mean(-rotate_C_proj_all[s25edit:e75edit+1, 2])

    ax2.set_ylim([center_plotz - 0.05, center_plotz + 0.05])
    ax2.set_xlim([center_ploty - 0.15, center_ploty + 0.15])
    ax2.grid(True)
    fig.suptitle('COM for '+ title, fontsize=20)    
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','COM for '+ title+ '.png'))

    #plt.show()
    # Save the figure
    # = 'path/to/post_analysis'  # Update with your path
    #plt.savefig(f'{post_analysis_dir}/COM.png')
    
    plt.close(fig)
    
    # Applying lowpass filter
    if direction > 0:
        rawyR = rotate_C_proj_all[:, 2]
    else:
        rawyR = -rotate_C_proj_all[:, 2]

    # Design and apply the filter

    fs = 30  # Sampling frequency
    cutoff_freq = 1  # Cutoff frequency
    steepness = 0.5  # Steepness
    # Design and apply the filter
    yR_filter = lowpass_filter(rawyR, cutoff_freq, fs,steepness)
    # Plotting the filtered data


    # Envelope detection
    from scipy.signal import hilbert

    analytic_signal = hilbert(yR_filter)
    amplitude_envelope = np.abs(analytic_signal)


    path = np.mean(np.vstack((amplitude_envelope, -amplitude_envelope)), axis=0)


    # Finding peaks and local minima
    peaks, _ = find_peaks(yR_filter)
    local_mins = islocalmin(yR_filter)
    min_locs = np.where(local_mins)[0]
    if min(min_locs) ==0:
        min_locs = np.delete(min_locs,0)
    if max(min_locs) ==len(yR_filter)-1:
        min_locs = np.delete(min_locs,-1)
    if min(peaks) ==0:
        peaks = np.delete(peaks,0)
    if max(peaks) ==len(yR_filter)-1:
        peaks = np.delete(peaks,-1)
    
    # Plot peaks and local minima


    # Finding the sorted extreme points
    mins = np.column_stack((min_locs, np.zeros(len(min_locs))))
    maxs = np.column_stack((peaks, np.ones(len(peaks))))
    extreme_points = np.vstack((mins, maxs))
    extreme_points = extreme_points[extreme_points[:, 0].argsort()]
    
    #import pdb;pdb.set_trace()
    # Adding start and end points
    
    sort_s = np.vstack(([0, -1], extreme_points))
    sort_e = np.vstack((extreme_points, [len(yR_filter)-1, -1]))
    

    # Interpolation
    
    dots_interpolation = (yR_filter[sort_s[:, 0].astype(int)] + yR_filter[sort_e[:, 0].astype(int)]) / 2
    segs = np.column_stack((sort_s[:, 0].astype(int), sort_e[:, 0].astype(int)))
    final_int = []
    #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    for i in range(len(dots_interpolation)):
        segment = yR_filter[segs[i, 0]:segs[i, 1]]
        k = np.abs(segment - dots_interpolation[i]).argmin()
        final_int.append([k + segs[i, 0], yR_filter[k + segs[i, 0]]])

    final_int = np.array(final_int)
    tck = splrep(final_int[:, 0], final_int[:, 1], s=0)
    # Interpolate using the fitted spline
    vq = splev(np.arange(1, len(yR_filter) + 1), tck)

    # Plot interpolated values
    plt.figure(figsize=(15, 9))
    plt.plot(rawyR, label='Raw')
    plt.plot(yR_filter, label='Filtered')
    plt.scatter(peaks, yR_filter[peaks], label='Peaks')
    plt.scatter(min_locs, yR_filter[min_locs], label='Local Minima')
    #plt.title(f'Patient {patient_aim}')
    plt.legend()
    plt.grid(True)
    plt.plot(vq, label='Interpolated')
    plt.plot(rawyR - vq, label='Calibrated Raw')
    plt.plot(yR_filter - vq, label='Calibrated Filtered')
    plt.legend()
    plt.title('calibrated COM on xy ' + title,fontsize =20)
    plt.grid(True)
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','calibrated COM on xy ' + title+ '.png'))
    #plt.show()

    # Perform calibration
    calibrate_yR_raw = rawyR - vq
    calibrate_yR_filter = yR_filter - vq
    # Create a new figure with specified size
     # Adjust the size as needed

    # Select data based on the direction
    if direction > 0:
        xR = rotate_C_proj_all[s25edit:e75edit+1, 0]
        yR = calibrate_yR_filter[s25edit:e75edit+1]
        zR = rotate_C_proj_all[s25edit:e75edit+1, 1]
    else:
        xR = rotate_C_proj_all[s25edit:e75edit+1, 0]
        yR = calibrate_yR_filter[s25edit:e75edit+1]
        zR = rotate_C_proj_all[s25edit:e75edit+1, 1]
    TR_mean = np.mean(yR)
    TR_std = np.std(yR)

    # Create a new figure
    #plt.figure(figsize=(15, 9))  # Adjust the size as needed

    # Find peaks

    locs,a = find_peaks(yR)
    pks = yR[locs]
    # Find local minima
    TF = argrelextrema(yR, np.less)[0]

    x = np.arange(0, len(yR) )
    temp = yR

    # Plotting


    temp = TF

    # Recheck for zY
    recheck = np.zeros((len(temp) + len(locs), 2))
    recheck[:len(temp), 0] = temp
    recheck[:len(temp), 1] = 0
    recheck[len(temp):, 0] = locs
    recheck[len(temp):, 1] = 1

    # Sort recheck
    recheck = recheck[np.argsort(recheck[:, 0])]

    # Filter recheck
    a = np.abs(np.diff(yR[recheck[:, 0].astype(int)]))
    recheck = recheck[np.insert(a >= 0.005, 0, True)]

    temp = []
    locs = []

    for recheck_index in range(len(recheck) - 1):
        if recheck[recheck_index, 1] != recheck[recheck_index + 1, 1]:
            if recheck[recheck_index, 1] == 0:
                temp.append(recheck[recheck_index, 0])
            elif recheck[recheck_index, 1] == 1:
                locs.append(recheck[recheck_index, 0])

    if recheck[-1, 1] == 0 and recheck[-1, 1] != recheck[-2, 1]:
        temp.append(recheck[-1, 0])
    elif recheck[-1, 1] == 1 and recheck[-1, 1] != recheck[-2, 1]:
        locs.append(recheck[-1, 0])

    # Filter temp and locs
    temp = np.array(temp)
    locs = np.array(locs)

    temp = temp[temp > locs[0]]

    ymin = int(temp[0])
    ymax = int(locs[0])
    #import pdb;pdb.set_trace()
    # Calibration with filtered data
    TF = argrelextrema(calibrate_yR_filter, np.less)[0]
    temp = TF
    locs, _ = find_peaks(calibrate_yR_filter)
    # Recheck for calibrated data
    recheck = np.zeros((len(temp) + len(locs), 2))
    recheck[:len(temp), 0] = temp
    recheck[:len(temp), 1] = 0
    recheck[len(temp):, 0] = locs
    recheck[len(temp):, 1] = 1

    # Sort recheck
    recheck = recheck[np.argsort(recheck[:, 0])]

    # Filter recheck
    a = np.abs(np.diff(calibrate_yR_filter[recheck[:, 0].astype(int)]))
    recheck = recheck[np.insert(a >= 0.005, 0, True)]

    temp = []
    locs = []

    for recheck_index in range(len(recheck) - 1):
        if recheck[recheck_index, 1] != recheck[recheck_index + 1, 1]:
            if recheck[recheck_index, 1] == 0:
                temp.append(recheck[recheck_index, 0])
            elif recheck[recheck_index, 1] == 1:
                locs.append(recheck[recheck_index, 0])

    if recheck[-1, 1] == 0 and recheck[-1, 1] != recheck[-2, 1]:
        temp.append(recheck[-1, 0])
    elif recheck[-1, 1] == 1 and recheck[-1, 1] != recheck[-2, 1]:
        locs.append(recheck[-1, 0])

    # Filter temp and locs
    temp = np.array(temp)
    locs = np.array(locs)
    temp = temp[temp < ymax + s25edit - 1]
    ymin0 = int(temp[-1])

    temp = locs[locs > ymin + s25edit - 1]
    ymax3 = int(temp[0])

    # # Plot calibrated filter
    # plt.figure(figsize=(15, 9))
    # plt.plot(calibrate_yR_filter, label='Calibrated yR Filter')
    # plt.scatter([ymin + s25edit , ymin0], calibrate_yR_filter[[ymin + s25edit , ymin0]], c='g', s=100, label='Minima Points')
    # plt.scatter([ymax + s25edit , ymax3], calibrate_yR_filter[[ymax + s25edit , ymax3]], c='r', s=100, label='Maxima Points')
    # plt.xlim([ymin0, ymax3])
    # plt.legend()
    # plt.show()

    # Compute middle indices
    middle1 = ymin0 + np.where(calibrate_yR_filter[ymin0:ymax + s25edit + 2] - np.mean([calibrate_yR_filter[ymin0], calibrate_yR_filter[ymax + s25edit]]) > 0)[0]
    middle2 = ymax + s25edit  + np.where(calibrate_yR_filter[ymax + s25edit:ymin + s25edit + 2] - np.mean([calibrate_yR_filter[ymax + s25edit], calibrate_yR_filter[ymin + s25edit]]) < 0)[0]
    middle3 = ymin + s25edit  + np.where(calibrate_yR_filter[ymin + s25edit:ymax3 + 2] - np.mean([calibrate_yR_filter[ymin + s25edit], calibrate_yR_filter[ymax3]]) > 0)[0]

    # Ensure that indices are within the array bounds
    middle1 = int(middle1[0]) if len(middle1) > 0 else ymin0
    middle2 = int(middle2[0]) if len(middle2) > 0 else ymax + s25edit - 1
    middle3 = int(middle3[0]) if len(middle3) > 0 else ymin + s25edit - 1

    # Compute lines
    line1 = np.arange(middle1, middle2 + 1)
    line2 = np.arange(middle2, middle3 + 1)
    #import pdb;pdb.set_trace()
    # Plot
    plt.figure(figsize=(15, 9))
    plt.plot(calibrate_yR_filter, label='Calibrated yR Filter')
    plt.plot(line1,np.linspace(calibrate_yR_filter[middle1], calibrate_yR_filter[middle2], len(line1)))
    plt.plot(line2,np.linspace(calibrate_yR_filter[middle2], calibrate_yR_filter[middle3], len(line2)))
    plt.scatter([ymin + s25edit , ymin0], calibrate_yR_filter[[ymin + s25edit , ymin0]],color='#00FF00', s=100, label='Minima Points')
    plt.scatter([ymax + s25edit , ymax3], calibrate_yR_filter[[ymax + s25edit , ymax3]], c='r', s=100, label='Maxima Points')
    plt.xlim([ymin0, ymax3])
    plt.fill_between(line1, calibrate_yR_filter[line1], np.linspace(calibrate_yR_filter[middle1], calibrate_yR_filter[middle2], len(line1)), color='k', alpha=0.1)
    plt.fill_between(line2, calibrate_yR_filter[line2], np.linspace(calibrate_yR_filter[middle2], calibrate_yR_filter[middle3], len(line2)), color='k', alpha=0.1)
    plt.legend()
    plt.grid(True)
    plt.title('COM lateral for ' + title ,fontsize = 20)
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','COM lateral for ' + title + '.png'))

    #plt.show()
    # Compute areas
    right_area = np.sum(calibrate_yR_filter[line1] - np.linspace(calibrate_yR_filter[middle1], calibrate_yR_filter[middle2], len(line1)))
    left_area = np.sum(np.linspace(calibrate_yR_filter[middle2], calibrate_yR_filter[middle3], len(line2)) - calibrate_yR_filter[line2])
    
    print(f"Right Area: {right_area}")
    print(f"Left Area: {left_area}")




    # Assuming yR, zR, s25edit, e75edit, rotate_C_proj_all, post_anlaysis_dir, patient_aim, and All are already defined

    # Create a new figure
    plt.figure(figsize=(15,9))  # Equivalent to 'normalized', 'outerposition', [0 0 1 1]
    #import pdb;pdb.set_trace()
    # Subplot 1
    plt.subplot(2, 1, 1)
    locs, _ = find_peaks(yR)
    locs = locs.astype(int)

    plt.xlim([0, len(zR)])
    plt.plot(yR)
    plt.scatter(locs, yR[locs], marker='v', c='k', label='Peaks')

    TF = argrelextrema(yR, np.less)[0]
    x = np.arange(len(yR))

    temp = TF

    # Recheck for zY
    recheck = np.vstack((np.column_stack((temp, np.zeros(len(temp)))), np.column_stack((locs, np.ones(len(locs))))))
    recheck = recheck[np.argsort(recheck[:, 0])]

    a = np.abs(np.diff(yR[recheck[:, 0].astype(int)]))
    recheck = recheck[np.insert(a >= 0.001, 0, True)]

    temp = []
    locs = []

    for recheck_index in range(len(recheck) - 1):
        if recheck[recheck_index, 1] != recheck[recheck_index + 1, 1]:
            if recheck[recheck_index, 1] == 0:
                temp.append(recheck[recheck_index, 0])
            elif recheck[recheck_index, 1] == 1:
                locs.append(recheck[recheck_index, 0])

    if recheck[-1, 1] == 0 and recheck[-1, 1] != recheck[-2, 1]:
        temp.append(recheck[-1, 0])
    elif recheck[-1, 1] == 1 and recheck[-1, 1] != recheck[-2, 1]:
        locs.append(recheck[-1, 0])

    temp = np.array(temp)
    locs = np.array(locs)
    locs = locs.astype(int)
    temp = temp.astype(int)
    # Plot updated local minima and maxima
    plt.scatter(locs, yR[locs], marker='v', c='b', label='Updated Peaks')
    plt.scatter(temp, yR[temp], c='r', marker='*', label='Updated Minima')
    plt.grid(True)
    plt.legend()

    temp_min_first = temp[temp < locs[0]]
    temp = temp[temp > locs[0]]

    if len(temp_min_first) != 0:
        temp_min_first = temp_min_first[-1]
    else:
        temp_min_first = 1

    ymin = temp[0]
    ymax = locs[0]
    if len(locs) > 1:
        ymax2 = locs[1]
    else:
        ymax2 = -1

    # Subplot 2
    plt.subplot(2, 1, 2)
    locs,_ = find_peaks(zR)

    TF = argrelextrema(zR, np.less)[0]
    x = np.arange(len(zR))
    temp = zR

    ll = locs
    pl = zR[locs]

    temp_min = TF
    zyR = np.argmin(np.abs(locs - ymax))
    zyL = np.argmin(np.abs(locs - ymin))

    temp_min_loc = temp_min[(temp_min - locs[zyR] <= 0) & (temp_min - temp_min_first + 2 >= 0)]
    if len(temp_min_loc) != 0:
        m1 = temp_min_loc[np.argmin(zR[temp_min_loc])]
    else:
        m1 = 1

    temp_min_loc = temp_min[(temp_min - locs[zyL] <= 0) & (temp_min - ymax + 2 >= 0)]
    if len(temp_min_loc) != 0:
        m2 = temp_min_loc[np.argmin(zR[temp_min_loc])]
    else:
        m2 = ymax

    if ymax2 == -1:
        min_next_locmin = np.argmin(zR[ymin:]) + ymin
    else:
        min_next_locmin = np.argmin(zR[ymin:ymax2]) + ymin

    temp1 = locs[(locs < m2) & (locs > m1)]
    temp2 = locs[(locs < min_next_locmin) & (locs > m2)]

    if len(temp1) > 0:
        temp_max_loc_R = temp1[np.argmax(zR[temp1])]
    else:
        temp_max_loc_R = np.argmin(np.abs(locs - ymax))

    if len(temp2) > 0:
        temp_max_loc_L = temp2[np.argmax(zR[temp2])]
    else:
        temp_max_loc_L = np.argmin(np.abs(locs - ymin))

    zyR = np.where(locs == temp_max_loc_R)[0][0]
    zyL = np.where(locs == temp_max_loc_L)[0][0]

    plt.plot(zR)
    plt.xlim([0, len(zR)])
    plt.scatter(locs[[zyR, zyL]], zR[locs[[zyR, zyL]]], color='#00FF00', s=100, label='Maxima')
    plt.scatter([m1, m2], zR[[m1, m2]], c='r', s=100, label='Minima')
    plt.legend()
    plt.grid(True)
    plt.suptitle('COM vertical '+ title, fontsize=20)
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','COM vertical '+ title+ '.png'))

    #plt.show()
    
    # Save the figure
    # plt.savefig(os.path.join(post_anlaysis_dir, 'COM_vertical.png'))

    # # Assuming All is a dictionary or list
    # All[patient_aim][60] = [zR[locs[zyR]], zR[m1]]
    # All[patient_aim][61] = [zR[locs[zyL]], zR[m2]]

    # Ensure All is saved or used as needed
    return  right_area,left_area,zR[locs[zyR]],zR[m1],zR[locs[zyL]],zR[m2]
def Speed_analysis(vcm30,frame_R_heel_lground,frame_L_heel_lground ,dir_task,title):

    # Create a new figure
    plt.figure(figsize=(15, 9))  # Equivalent to 'normalized', 'outerposition', [0 0 1 1]

    # Fill outliers in speed30_cm
    speed30_cm = np.array(vcm30)
    # Calculate the 0th and 95th percentiles
    lower_bound, upper_bound = np.percentile(speed30_cm, [0, 95])

    # Identify outliers
    outliers = (speed30_cm < lower_bound) | (speed30_cm > upper_bound)

    # Create a copy of the original array
    B = speed30_cm.copy()

    # Replace outliers with NaN
    B[outliers] = np.nan

    # Fill NaN values using linear interpolation
    B = pd.Series(B).interpolate(method='linear').values

    # Plot B
    plt.plot(B, linewidth=1.5)


    # Calculate speed_midpoint and mean_gap
    speed_midpoint = np.sort(np.concatenate([frame_R_heel_lground, frame_L_heel_lground]))
    mean_gap = int(round_half_up(np.mean(np.diff(np.sort(np.concatenate([frame_R_heel_lground, frame_L_heel_lground]))))))

    while speed_midpoint[0] - mean_gap > 1:
        speed_midpoint = np.insert(speed_midpoint, 0, speed_midpoint[0] - mean_gap)

    while speed_midpoint[-1] + mean_gap < len(B):
        speed_midpoint = np.append(speed_midpoint, speed_midpoint[-1] + mean_gap)

    speed_midpoint = np.concatenate(([0], speed_midpoint, [len(B)-1]))
    interp_speed = []

    for i in range(len(speed_midpoint) - 2):
        temp_interp = np.sort(B[speed_midpoint[i]:speed_midpoint[i+2]])
        if B[speed_midpoint[i]] > B[speed_midpoint[i+2]]:
            temp_interp = np.flipud(temp_interp)

        interp_speed.append([speed_midpoint[i+1], temp_interp[speed_midpoint[i+1] - speed_midpoint[i]]])

    interp_speed = np.vstack(([[speed_midpoint[0], B[speed_midpoint[0]]]], interp_speed, [[speed_midpoint[-1], B[speed_midpoint[-1]]]]))
    interp_speed = np.array(interp_speed)  # Ensure it is a numpy array

    # Interpolate mean_velocity using PCHIP
    f = PchipInterpolator(interp_speed[:, 0], interp_speed[:, 1])
    mean_velocity = f(np.arange(0, len(B) ))

    flunc_velocity = B - mean_velocity

    # Plot results
    #plt.axvline(x=speed_midpoint, color='k', linestyle='--')
    plt.plot(mean_velocity, linewidth=1.5, label='Mean Velocity')
    plt.plot(flunc_velocity, linewidth=1.5, label='Fluctuation Velocity')
    plt.plot(B, linewidth=1.5, label='Raw Data')
    plt.scatter(interp_speed[:, 0], interp_speed[:, 1])

    TR_steady = 0.7  # 70% above define as steady
    TR_startend = 0.2  # 20% above define as start walking or end walking

    plt.close('all')
    plt.figure(figsize=(15, 9))  # Equivalent to 'normalized', 'outerposition', [0 0 1 1])

    plt.plot(mean_velocity, linewidth=1.5, label='Mean Velocity')
    plt.plot(flunc_velocity, linewidth=1.5, label='Fluctuation Velocity')
    plt.plot(B, linewidth=1.5, label='Raw Data',color='#CCCC00')

    value = np.max(mean_velocity)
    loc = np.argmax(mean_velocity)
    bound = []

    for check in range(1, len(mean_velocity)):
        if mean_velocity[check] > TR_steady * (value - np.min(mean_velocity)) + np.min(mean_velocity):
            bound.append(check)

    bound_start_end = []

    for check in range(1, len(mean_velocity)):
        if mean_velocity[check] > TR_startend * (value - np.min(mean_velocity)) + np.min(mean_velocity):
            bound_start_end.append(check)

    rms_final_steady = np.sqrt(np.mean(flunc_velocity[min(bound):max(bound)]**2))
    rms_start_end = np.sqrt(np.mean(flunc_velocity[np.concatenate((np.arange(min(bound_start_end), min(bound)), np.arange(max(bound), max(bound_start_end))))]**2))
    rms_All = np.sqrt(np.mean(flunc_velocity[min(bound_start_end):max(bound_start_end)]**2))
    #import pdb;pdb.set_trace()
    plt.axvline(x=[min(bound)], linestyle='-.', color='#00FF00', linewidth=1.5)
    plt.axvline(x=[max(bound)], linestyle='-.', color='#00FF00', linewidth=1.5)
    plt.axvline(x=[min(bound_start_end)], linestyle='-.',color='dimgrey', linewidth=1.5)
    plt.axvline(x=[max(bound_start_end)], linestyle='-.',color='dimgrey', linewidth=1.5)

    plt.text(0.05, 0.9, f'rms steady = {rms_final_steady:.5f}', transform=plt.gca().transAxes, fontsize=20)
    plt.text(0.05, 0.8, f'rms start end = {rms_start_end:.5f}', transform=plt.gca().transAxes, fontsize=20)
    plt.text(0.05, 0.7, f'rms All = {rms_All:.5f}', transform=plt.gca().transAxes, fontsize=20)
    plt.text(0.05, 0.6, f'max speed = {np.max(mean_velocity):.5f}', transform=plt.gca().transAxes, fontsize=20)
    plt.title('Speed '+ title,fontsize = 20)
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the figure
    #plt.savefig(os.path.join(post_anlaysis_dir, 'COM_vertical.png'))
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','Speed '+ title+ '.png'))
    #plt.show()
    return rms_final_steady,rms_start_end,rms_All,np.max(mean_velocity)
def stride_length(data,frame_R_heel_sground,frame_L_heel_sground,dir_task,title):
    
    R_heel = data[:, body_parts['RHeel']:body_parts['RHeel']+3]
    x = R_heel[:, 0]
    y = R_heel[:, 2]
    xr = x
    yr = y
    x1 = np.roll(x, 1)
    y1 = np.roll(y, 1)
    x1[0] = x[0]
    y1[0] = y[0]

    vx = x - x1
    vy = y - y1

    v = np.sqrt(vx**2 + vy**2) * SR

    # Identify local minima
    TF = np.r_[True, v[1:] < v[:-1]] & np.r_[v[:-1] < v[1:], True]
    loc = np.where(TF)[0]
    lr = loc
    num = np.arange(1, len(loc) + 1)
    minima = v[loc]
    pr = minima

    # plt.figure(figsize=(15, 9))
    # plt.subplot(2, 1, 1)
    # plt.plot(v)
    # plt.scatter(loc, minima, marker="^", color='g', s=150)
    # for i, txt in enumerate(num):
    #     plt.annotate(txt, (loc[i], minima[i]))
    # plt.grid()
    # plt.title('Speed of Right Heel in xy plane')
    L_heel = data[:, body_parts['LHeel']:body_parts['LHeel']+3]
    x = L_heel[:, 0]
    y = L_heel[:, 2]
    xl = x
    yl=y
    x1 = np.roll(x, 1)
    y1 = np.roll(y, 1)
    x1[0] = x[0]
    y1[0] = y[0]

    vx = x - x1
    vy = y - y1

    v = np.sqrt(vx**2 + vy**2) * SR

    # Identify local minima
    TF = np.r_[True, v[1:] < v[:-1]] & np.r_[v[:-1] < v[1:], True]
    loc = np.where(TF)[0]
    num = np.arange(1, len(loc) + 1)
    minima = v[loc]
    pl = minima

    # plt.subplot(2, 1, 2)
    # plt.plot(v)
    # plt.scatter(loc, minima, marker="^", color='g', s=150)
    # for i, txt in enumerate(num):
    #     plt.annotate(txt, (loc[i], minima[i]))
    # plt.grid()
    # plt.title('Speed of Left Heel in xy plane')
    # plt.show()
    xmr, ymr, xml, yml = [], [], [], []
    pace_r, pace_l = [], []

    for i in range(len(frame_R_heel_sground) - 1):
        pace_r.append(np.sqrt((xr[frame_R_heel_sground[i+1]] - xr[frame_R_heel_sground[i]])**2 + (yr[frame_R_heel_sground[i+1]] - yr[frame_R_heel_sground[i]])**2))
        xmr.append((xr[frame_R_heel_sground[i+1]] + xr[frame_R_heel_sground[i]]) / 2)
        ymr.append((yr[frame_R_heel_sground[i+1]] + yr[frame_R_heel_sground[i]]) / 2)

    for i in range(len(frame_L_heel_sground) - 1):
        pace_l.append(np.sqrt((xl[frame_L_heel_sground[i+1]] - xl[frame_L_heel_sground[i]])**2 + (yl[frame_L_heel_sground[i+1]] - yl[frame_L_heel_sground[i]])**2))
        xml.append((xl[frame_L_heel_sground[i+1]] + xl[frame_L_heel_sground[i]]) / 2)
        yml.append((yl[frame_L_heel_sground[i+1]] + yl[frame_L_heel_sground[i]]) / 2)

    mid_r_index = round_half_up((len(pace_r) + 1) / 2) - 1  # Adjust for zero-based indexing
    mid_l_index = round_half_up((len(pace_l) + 1) / 2) - 1  # Adjust for zero-based indexing

    # Creating the scatter plot

    # Assuming R_heel, L_heel, frame_R_heel_sground, frame_L_heel_sground, xmr, ymr, xml, yml, pace_r, pace_l are already defined

    plt.figure(figsize=(15, 9))  # Equivalent to 'units', 'normalized', 'outerposition', [0 0 1 1]

    # Scatter plots for heel traces
    plt.scatter(R_heel[:, 0], R_heel[:, 2], label='Right Heel Trace')
    plt.scatter(L_heel[:, 0], L_heel[:, 2], label='Left Heel Trace')

    # Scatter plots for heel strikes
    plt.scatter(R_heel[frame_R_heel_sground, 0], R_heel[frame_R_heel_sground, 2], color='r', s=100, label='Right Heel Strikes')
    plt.scatter(L_heel[frame_L_heel_sground, 0], L_heel[frame_L_heel_sground, 2], color='r', s=100, label='Left Heel Strikes')

    # Adding text annotations for pace_r and pace_l
    for i in range(len(xmr)):
        plt.text(xmr[i], ymr[i], f'{pace_r[i]:.4f}', fontsize=9)

    for i in range(len(xml)):
        plt.text(xml[i], yml[i], f'{pace_l[i]:.4f}', fontsize=9)

    plt.legend()
    plt.grid(True)
    plt.title('Stride length '+ title,fontsize = 20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')


    # Adding scatter points for the heel traces


    # Highlighting specific points
    plt.scatter([xmr[mid_r_index] - 0.01], [ymr[mid_r_index]], marker='v',color='#00FF00', s=100)
    plt.scatter([xml[mid_l_index] - 0.01], [yml[mid_l_index]], marker='v',color='#00FF00', s=100)

    # Adding legend and title
    plt.legend()
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','Stride length '+ title+ '.png'))
    # Show the plot
    #plt.show()

    # Initializing temp_r and temp_l arrays
    temp_r = np.zeros(len(pace_r))
    temp_l = np.zeros(len(pace_l))

    # Setting the middle value
    temp_r[mid_r_index] = pace_r[mid_r_index]
    temp_l[mid_l_index] = pace_l[mid_l_index]
    return pace_r,temp_r,pace_l,temp_l
def knee_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title):
    loc_max_finalR = []
    loc_min_finalR = []
    R_knee = angle['knee_angle_r'].to_numpy()
    L_knee = angle['knee_angle_l'].to_numpy()
    # Process Right Knee Data
    for temp in range(len(frame_R_heel_sground) - 1):
        if temp > len(frame_R_heel_lground) - 2:
            maxval = np.max(R_knee[frame_R_heel_sground[temp]:frame_R_heel_sground[temp+1]])
            locmax = np.argmax(R_knee[frame_R_heel_sground[temp]:frame_R_heel_sground[temp+1]])
            loc_max_finalR.append(locmax + frame_R_heel_sground[temp])
        else:
            maxval = np.max(R_knee[frame_R_heel_lground[temp]:frame_R_heel_lground[temp+1]])
            locmax = np.argmax(R_knee[frame_R_heel_lground[temp]:frame_R_heel_lground[temp+1]])
            loc_max_finalR.append(locmax + frame_R_heel_lground[temp])
        
        if temp == 0:
            minval = np.min(R_knee[frame_R_heel_lground[temp]:locmax + frame_R_heel_lground[temp]])
            locmin = np.argmin(R_knee[frame_R_heel_lground[temp]:locmax + frame_R_heel_lground[temp]])
            loc_min_finalR.append(locmin + frame_R_heel_lground[temp])
        else:
            minval = np.min(R_knee[loc_max_finalR[temp-1]:loc_max_finalR[temp]])
            locmin = np.argmin(R_knee[loc_max_finalR[temp-1]:loc_max_finalR[temp]])
            loc_min_finalR.append(locmin + loc_max_finalR[temp-1])

    # Initialize lists to store locations of max and min values for Left Knee Data
    loc_max_finalL = []
    loc_min_finalL = []

    # Process Left Knee Data
    for temp in range(len(frame_L_heel_sground) - 1):
        if temp > len(frame_L_heel_lground) - 2:
            maxval = np.max(L_knee[frame_L_heel_sground[temp]:frame_L_heel_sground[temp+1]])
            locmax = np.argmax(L_knee[frame_L_heel_sground[temp]:frame_L_heel_sground[temp+1]])
            loc_max_finalL.append(locmax + frame_L_heel_sground[temp])
        else:
            maxval = np.max(L_knee[frame_L_heel_lground[temp]:frame_L_heel_lground[temp+1]])
            locmax = np.argmax(L_knee[frame_L_heel_lground[temp]:frame_L_heel_lground[temp+1]])
            loc_max_finalL.append(locmax + frame_L_heel_lground[temp])
        
        if temp == 0:
            minval = np.min(L_knee[frame_L_heel_lground[temp]:locmax + frame_L_heel_lground[temp]+ 1])
            locmin = np.argmin(L_knee[frame_L_heel_lground[temp]:locmax + frame_L_heel_lground[temp]+ 1])
            loc_min_finalL.append(locmin + frame_L_heel_lground[temp])
        else:
            minval = np.min(L_knee[loc_max_finalL[temp-1]:loc_max_finalL[temp]])
            locmin = np.argmin(L_knee[loc_max_finalL[temp-1]:loc_max_finalL[temp]])
            loc_min_finalL.append(locmin + loc_max_finalL[temp-1])

    # Plotting
    plt.figure(figsize=(15, 9))

    # Plot Right Knee Data
    plt.plot(R_knee, linewidth=1.5, color='r', label='Right Knee')
    plt.scatter(loc_max_finalR, R_knee[loc_max_finalR], marker='v', color='#00FF00', s=150)
    plt.scatter([loc_max_finalR[round_half_up((len(loc_max_finalR) + 1) / 2) - 1]], [R_knee[loc_max_finalR[(round_half_up((len(loc_max_finalR) + 1) / 2) - 1)]]], marker='*', color='magenta',s=700)
    # Plot Left Knee Data
    plt.plot(L_knee, linewidth=1.5, color='b', label='Left Knee')
    plt.scatter(loc_max_finalL, L_knee[loc_max_finalL], marker='v', color='cyan', s=150)
    plt.scatter([loc_max_finalL[round_half_up((len(loc_max_finalL) + 1) / 2) - 1]], [L_knee[loc_max_finalL[(round_half_up((len(loc_max_finalL) + 1) / 2) - 1)]]], marker='*', color='magenta', s=700)
    plt.grid(True)
    plt.title('knee flexion '+ title,fontsize = 20)
    plt.legend()
    # plt.title(f'{png_name} knee', fontsize=20)
    # plt.savefig(os.path.join(post_analysis_dir, 'knee.png'))
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','knee flexion '+ title+ '.png'))

    #plt.show()
    return  [R_knee[loc_max_finalR[(round_half_up((len(loc_max_finalR) + 1) / 2) - 1)]]],[L_knee[loc_max_finalL[(round_half_up((len(loc_max_finalL) + 1) / 2) - 1)]]],R_knee[loc_max_finalR],L_knee[loc_max_finalL]
def ankle_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title):
# Sample data for demonstration
    R_ankle = angle['ankle_angle_r'].to_numpy()
    L_ankle = angle['ankle_angle_l'].to_numpy()

    # Adjust L_ankle values if needed
    if np.median(L_ankle) < -100:
        L_ankle += 360

    # Adjust R_ankle values if needed
    if np.median(R_ankle) < -100:
        R_ankle = R_ankle  # This line does nothing, so it's omitted in Python

    # Initialize lists to store the results
    loc_max_finalR = []
    loc_min_finalR = []
    loc_max_finalL = []
    loc_min_finalL = []

    # Process Right Ankle
    for temp in range(len(frame_R_heel_sground) - 1):
        maxval = np.max(R_ankle[frame_R_heel_sground[temp]:frame_R_heel_sground[temp+1]])
        locmax = np.argmax(R_ankle[frame_R_heel_sground[temp]:frame_R_heel_sground[temp+1]])
        loc_max_finalR.append(locmax + frame_R_heel_sground[temp])
        
        if temp == 0:
            minval = np.min(R_ankle[frame_R_heel_sground[temp]:locmax + frame_R_heel_sground[temp] + 1])
            locmin = np.argmin(R_ankle[frame_R_heel_sground[temp]:locmax + frame_R_heel_sground[temp] + 1])
            loc_min_finalR.append(locmin + frame_R_heel_sground[temp])
        else:
            minval = np.min(R_ankle[loc_max_finalR[temp-1]:loc_max_finalR[temp]])
            locmin = np.argmin(R_ankle[loc_max_finalR[temp-1]:loc_max_finalR[temp]])
            loc_min_finalR.append(locmin + loc_max_finalR[temp-1])

    # Process Left Ankle
    for temp in range(len(frame_L_heel_sground) - 1):
        maxval = np.max(L_ankle[frame_L_heel_sground[temp]:frame_L_heel_sground[temp+1]])
        locmax = np.argmax(L_ankle[frame_L_heel_sground[temp]:frame_L_heel_sground[temp+1]])
        loc_max_finalL.append(locmax + frame_L_heel_sground[temp])
        
        if temp == 0:
            minval = np.min(R_ankle[frame_L_heel_sground[temp]:locmax + frame_L_heel_sground[temp] + 1])
            locmin = np.argmin(R_ankle[frame_L_heel_sground[temp]:locmax + frame_L_heel_sground[temp] + 1])
            loc_min_finalL.append(locmin + frame_L_heel_sground[temp])
        else:
            minval = np.min(L_ankle[loc_max_finalL[temp-1]:loc_max_finalL[temp]])
            locmin = np.argmin(L_ankle[loc_max_finalL[temp-1]:loc_max_finalL[temp]])
            loc_min_finalL.append(locmin + loc_max_finalL[temp-1])

    # Plotting
    plt.figure(figsize=(15, 9))
    plt.grid(True)
    plt.plot(R_ankle, linewidth=1.5, color='r', label='Right Ankle')
    plt.plot(L_ankle, linewidth=1.5, color='b', label='Left Ankle')

    # Scatter plots
    plt.scatter(loc_max_finalR, R_ankle[loc_max_finalR], marker='v', color='#00FF00', s=150)
    plt.scatter([loc_max_finalR[round_half_up((len(loc_max_finalR) + 1) / 2) - 1]], [R_ankle[loc_max_finalR[(round_half_up((len(loc_max_finalR) + 1) / 2)) - 1]]], marker='*', color='magenta', s=700)

    plt.scatter(loc_max_finalL, L_ankle[loc_max_finalL], marker='v', color='cyan', s=150)
    plt.scatter([loc_max_finalL[round_half_up((len(loc_max_finalL) + 1) / 2) - 1]], [L_ankle[loc_max_finalL[(round_half_up((len(loc_max_finalL) + 1) / 2)) - 1]]], marker='*', color='magenta', s=700)

    plt.legend()
    
    plt.title('ankle dorsiflexion '+ title,fontsize = 20)
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','ankle dorsiflexion '+ title+ '.png'))

    #plt.show()
    return [R_ankle[loc_max_finalR[(round_half_up((len(loc_max_finalR) + 1) / 2)) - 1]]],[L_ankle[loc_max_finalL[(round_half_up((len(loc_max_finalL) + 1) / 2)) - 1]]],R_ankle[loc_max_finalR], L_ankle[loc_max_finalL]
def hip_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title):
    R_Hip = angle['hip_flexion_r'].to_numpy()
    L_Hip = angle['hip_flexion_l'].to_numpy()

    loc_max_finalR = []
    loc_min_finalR = []
    loc_max_finalL = []
    loc_min_finalL = []

    # Process Right Hip
    for temp in range(len(frame_R_heel_sground) - 1):
        # Detect local maxima
        peaks, _ = find_peaks(R_Hip)

    # Create an array to store local maxima only
        R_Hip_locmaxonly = np.zeros_like(R_Hip)
        R_Hip_locmaxonly[peaks] = R_Hip[peaks]    
        if temp > len(frame_R_heel_lground) - 2:
            maxval = np.max(R_Hip_locmaxonly[frame_R_heel_sground[temp]:frame_R_heel_sground[temp + 1]])
            locmax = np.argmax(R_Hip_locmaxonly[frame_R_heel_sground[temp]:frame_R_heel_sground[temp + 1]])
            loc_max_finalR.append(locmax + frame_R_heel_sground[temp])
        else:
            maxval = np.max(R_Hip_locmaxonly[frame_R_heel_lground[temp]:frame_R_heel_lground[temp + 1]])
            locmax = np.argmax(R_Hip_locmaxonly[frame_R_heel_lground[temp]:frame_R_heel_lground[temp + 1]])
            loc_max_finalR.append(locmax + frame_R_heel_lground[temp])
            
        if temp == 0:
            minval = np.min(R_Hip[frame_R_heel_lground[temp]:locmax + frame_R_heel_lground[temp]])
            locmin = np.argmin(R_Hip[frame_R_heel_lground[temp]:locmax + frame_R_heel_lground[temp]])
            loc_min_finalR.append(locmin + frame_R_heel_lground[temp])
        else:
            minval = np.min(R_Hip[loc_max_finalR[temp - 1]:loc_max_finalR[temp]])
            locmin = np.argmin(R_Hip[loc_max_finalR[temp - 1]:loc_max_finalR[temp]])
            loc_min_finalR.append(locmin + loc_max_finalR[temp - 1])

    # Process Left Hip
    for temp in range(len(frame_L_heel_sground) - 1):
        # Detect local maxima
        peaks, _ = find_peaks(L_Hip)

            # Create an array to store local maxima only
        L_Hip_locmaxonly = np.zeros_like(L_Hip)
        L_Hip_locmaxonly[peaks] = L_Hip[peaks]      
        if temp > len(frame_L_heel_lground) - 2:
            maxval = np.max(L_Hip_locmaxonly[frame_L_heel_sground[temp]:frame_L_heel_sground[temp + 1]])
            locmax = np.argmax(L_Hip_locmaxonly[frame_L_heel_sground[temp]:frame_L_heel_sground[temp + 1]])
            loc_max_finalL.append(locmax + frame_L_heel_sground[temp])
        else:
            maxval = np.max(L_Hip_locmaxonly[frame_L_heel_lground[temp]:frame_L_heel_lground[temp + 1]])
            locmax = np.argmax(L_Hip_locmaxonly[frame_L_heel_lground[temp]:frame_L_heel_lground[temp + 1]])
            loc_max_finalL.append(locmax + frame_L_heel_lground[temp])
            
        if temp == 0:
            minval = np.min(L_Hip[frame_L_heel_lground[temp]:locmax + frame_L_heel_lground[temp]])
            locmin = np.argmin(L_Hip[frame_L_heel_lground[temp]:locmax + frame_L_heel_lground[temp]])
            loc_min_finalL.append(locmin + frame_L_heel_lground[temp])
        else:
            minval = np.min(L_Hip[loc_max_finalL[temp - 1]:loc_max_finalL[temp]])
            locmin = np.argmin(L_Hip[loc_max_finalL[temp - 1]:loc_max_finalL[temp]])
            loc_min_finalL.append(locmin + loc_max_finalL[temp - 1])
    # Plotting
    plt.figure(figsize=(15, 9))
    plt.plot(R_Hip, linewidth=1.5, color='r', label='Right Hip')
    plt.plot(L_Hip, linewidth=1.5, color='b', label='Left Hip')

    # Scatter plots
    plt.scatter(loc_max_finalR, R_Hip[loc_max_finalR], marker='v', color='#00FF00', s=150)
    plt.scatter([loc_max_finalR[round_half_up((len(loc_max_finalR) + 1) / 2) - 1]], [R_Hip[loc_max_finalR[round_half_up((len(loc_max_finalR) + 1) / 2) - 1]]], marker='*', color='magenta', s=700)

    plt.scatter(loc_max_finalL, L_Hip[loc_max_finalL], marker='v', color='cyan', s=150)
    plt.scatter([loc_max_finalL[round_half_up((len(loc_max_finalL) + 1) / 2) - 1]], [L_Hip[loc_max_finalL[round_half_up((len(loc_max_finalL) + 1) / 2) - 1]]],marker='*', color='magenta', s=700)
    plt.title('Hip Angles')
    plt.legend()
    plt.grid(True)
    plt.title('hip flexion '+ title,fontsize = 20)
    if title !='False':
        plt.savefig(os.path.join(dir_task,'post_analysis','hip flexion '+ title+ '.png'))
    #import pdb;pdb.set_trace()
    #plt.show()
    return [R_Hip[loc_max_finalR[round_half_up((len(loc_max_finalR) + 1) / 2) - 1]]],[L_Hip[loc_max_finalL[round_half_up((len(loc_max_finalL) + 1) / 2) - 1]]],R_Hip[loc_max_finalR], L_Hip[loc_max_finalL]
def excel_output(dir_task,patient_id,date_str,task_str,R_hip_steady,L_hip_steady,R_knee_steady,L_knee_steady,R_ankle_steady,L_ankle_steady,max_mean_velocity,rms_final_steady,rms_start_end,rms_All,AUC_R,AUC_L,vertical_maxR,vertical_minR,vertical_maxL,vertical_minL,temp_r,temp_l):
    excel_output = {
    'Title': [  'Tasktype',
                'Version', 
                'Patient ID',
                'Task Date',
                'Task Name',
                'Maximum Hip Angle Right in steady state',
                'Maximum Hip Angle Left in steady state',
                'Maximum Knee Angle Right in steady state',
                'Maximum Knee Angle Left in steady state',
                'Maximum Ankle Angle Right in steady state',
                'Maximum Ankle Angle Left in steady state',
                'Maximum Mean Speed',
                'Variability of speed on steady state (rms)',
                'Variability of speed on Acceleration and Decceleration state (rms)',
                'Variability of speed on All state (rms)',
                'Center of Mass of Lateral direction for AUC on Right side in steady state',
                'Center of Mass of Lateral direction for AUC on Left side in steady state',
                'Center of Mass of vertical direction for distance swings at Right side in steady state',
                'Center of Mass of vertical direction for distance swings at Left side in steady state',
                'Stride lenght Right in steady state',
                'Stride length Left in steady state'],
    'Value': [  Tasktype,
                Version,
                patient_id,
                date_str,
                task_str,
                R_hip_steady[0],
                L_hip_steady[0],
                R_knee_steady[0],
                L_knee_steady[0],
                R_ankle_steady[0],
                L_ankle_steady[0],
                max_mean_velocity,
                rms_final_steady,
                rms_start_end,
                rms_All,
                AUC_R,
                AUC_L,
                vertical_maxR-vertical_minR,
                vertical_maxL-vertical_minL,
                max(temp_r),
                max(temp_l)]
    }
    import os

    # Directory to search
    directory = os.path.join(dir_task,'post_analysis')

    # Initialize dictionaries to store categorized filenames
    file_categories = {
        'hip': [],
        'knee': [],
        'ankle': [],
        'Speed': [],
        'COM': [],
        'COM lateral': [],
        'COM vertical': [],
        'Stride length': []
    }

    # Iterate through the files in the directory
    for f in os.listdir(directory):
        if f.endswith('.png'):
            if f.startswith('hip'):
                file_categories['hip'].append(f)
            elif f.startswith('knee'):
                file_categories['knee'].append(f)
            elif f.startswith('ankle'):
                file_categories['ankle'].append(f)
            elif f.startswith('Speed'):
                file_categories['Speed'].append(f)
            elif f.startswith('COM lateral'):
                file_categories['COM lateral'].append(f)
            elif f.startswith('COM vertical'):
                file_categories['COM vertical'].append(f)
            elif f.startswith('Stride length'):
                file_categories['Stride length'].append(f)
            elif f.startswith('COM'):
                file_categories['COM'].append(f)


    #import pdb;pdb.set_trace()
    excel_output_pd = pd.DataFrame(excel_output)
    excel_output_pd.to_excel(os.path.join(dir_task,'post_analysis','output.xlsx'), index=False) 
    # Convert to DataFrame


    # Write DataFrame to Excel file
    file_path = os.path.join(dir_task,'post_analysis','output.xlsx')
    excel_output_pd.to_excel(file_path, index=False, header=False)

    # Adjust the column widths to fit the content
    workbook = load_workbook(file_path)
    worksheet = workbook.active
    count =0
    # Print the categorized filenames
    for category, files in file_categories.items():
        
        print(f"{category.capitalize()} files: {files}")
        img = Image(os.path.join(dir_task,'post_analysis',files[0]))
        img.width, img.height =700,420
        img.anchor = 'D'+str(count*21+1)  # Position the image at cell A10
        worksheet.add_image(img)
        count = count+1
    


    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value)) for cell in column_cells if cell.value is not None)
        worksheet.column_dimensions[column_cells[0].column_letter].width = max_length + 2

    workbook.save(file_path)

def dict_output(R_hip_steady,L_hip_steady,R_knee_steady,L_knee_steady,R_ankle_steady,L_ankle_steady,max_mean_velocity,rms_final_steady,rms_start_end,rms_All,AUC_R,AUC_L,vertical_maxR,vertical_minR,vertical_maxL,vertical_minL,temp_r,temp_l):
    output = {
    'Tasktype': Tasktype,
    'Version' : Version,
    'Maximum Hip Angle Right in steady state':  R_hip_steady[0],
    'Maximum Hip Angle Left in steady state':L_hip_steady[0],
    'Maximum Knee Angle Right in steady state': R_knee_steady[0],
    'Maximum Knee Angle Left in steady state':L_knee_steady[0],
    'Maximum Ankle Angle Right in steady state':R_ankle_steady[0],
    'Maximum Ankle Angle Left in steady state':L_ankle_steady[0],
    'Maximum Mean Speed':max_mean_velocity,
    'Variability of speed on steady state (rms)': rms_final_steady,
    'Variability of speed on Acceleration and Decceleration state (rms)': rms_start_end,
    'Variability of speed on All state (rms)':rms_All,
    'Center of Mass of Lateral direction for AUC on Right side in steady state': AUC_R,
    'Center of Mass of Lateral direction for AUC on Left side in steady state': AUC_L,
    'Center of Mass of vertical direction for distance swings at Right side in steady state':vertical_maxR-vertical_minR,
    'Center of Mass of vertical direction for distance swings at Left side in steady state':vertical_maxL-vertical_minL,
    'Stride lenght Right in steady state':  max(temp_r),
    'Stride length Left in steady state': max(temp_l)
    }         
    return output

def gait1(dir_calculated):
    tasks=choose_file(dir_calculated)
    #
    for i in range(len(tasks)):
        dir_task = tasks[i]
        create_post_analysis_dir(dir_task)
        path_parts = dir_task.split(os.sep)
        patient_data_index = path_parts.index('Patient_data')
        patient_id = path_parts[patient_data_index + 1]
        date_str = path_parts[patient_data_index + 2]
        task_str = path_parts[patient_data_index + 4]
        title = patient_id+'_'+date_str+'_'+task_str
        
        IK_dir = os.path.join(dir_task, 'opensim', 'Balancing_for_IK_BODY.mot')
        trc_dir = os.path.join(dir_task, 'opensim', 'Empty_project_filt_0-30.trc')
        data,angle,cm ,vr30, vl30,vcm30,frame_R_heel_sground,frame_L_heel_sground,frame_R_heel_lground,frame_L_heel_lground =initial_read_data(IK_dir,trc_dir,dir_task,title)
        #import pdb;pdb.set_trace()
        AUC_R,AUC_L,vertical_maxR,vertical_minR,vertical_maxL,vertical_minL =COM_analysis(cm,frame_R_heel_sground,dir_task,title)
        rms_final_steady,rms_start_end,rms_All,max_mean_velocity=Speed_analysis(vcm30,frame_R_heel_lground,frame_L_heel_lground ,dir_task,title)
        pace_r,temp_r,pace_l,temp_l=stride_length(data,frame_R_heel_sground,frame_L_heel_sground,dir_task,title)
        R_knee_steady,L_knee_steady,R_knee,L_knee=knee_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
        R_ankle_steady,L_ankle_steady,R_ankle,L_ankle=ankle_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
        R_hip_steady,L_hip_steady,R_hip,L_hip=hip_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
        excel_output(dir_task,patient_id,date_str,task_str,R_hip_steady,L_hip_steady,R_knee_steady,L_knee_steady,R_ankle_steady,L_ankle_steady,max_mean_velocity,rms_final_steady,rms_start_end,rms_All,AUC_R,AUC_L,vertical_maxR,vertical_minR,vertical_maxL,vertical_minL,temp_r,temp_l)
def gait1_singlefile(IK_dir,trc_dir,output_dir,patient_id,date_str,task_str):  
    dir_task = output_dir
    create_post_analysis_dir(dir_task)
    path_parts = dir_task.split(os.sep)
    # patient_data_index = path_parts.index('Patient_data')
    # patient_id = path_parts[patient_data_index + 1]
    # date_str = path_parts[patient_data_index + 2]
    # task_str = path_parts[patient_data_index + 4]
    title = patient_id+'_'+date_str+'_'+task_str

    data,angle,cm ,vr30, vl30,vcm30,frame_R_heel_sground,frame_L_heel_sground,frame_R_heel_lground,frame_L_heel_lground =initial_read_data(IK_dir,trc_dir,dir_task,title)
    AUC_R,AUC_L,vertical_maxR,vertical_minR,vertical_maxL,vertical_minL =COM_analysis(cm,frame_R_heel_sground,dir_task,title)
    rms_final_steady,rms_start_end,rms_All,max_mean_velocity=Speed_analysis(vcm30,frame_R_heel_lground,frame_L_heel_lground ,dir_task,title)
    pace_r,temp_r,pace_l,temp_l=stride_length(data,frame_R_heel_sground,frame_L_heel_sground,dir_task,title)
    R_knee_steady,L_knee_steady,R_knee,L_knee=knee_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
    R_ankle_steady,L_ankle_steady,R_ankle,L_ankle=ankle_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
    R_hip_steady,L_hip_steady,R_hip,L_hip=hip_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
    excel_output(dir_task,patient_id,date_str,task_str,R_hip_steady,L_hip_steady,R_knee_steady,L_knee_steady,R_ankle_steady,L_ankle_steady,max_mean_velocity,rms_final_steady,rms_start_end,rms_All,AUC_R,AUC_L,vertical_maxR,vertical_minR,vertical_maxL,vertical_minL,temp_r,temp_l)
def gait1_dictoutput(IK_dir,trc_dir,output_dir):
    dir_task = output_dir
    path_parts = dir_task.split(os.sep)
    title = 'False'
    data,angle,cm ,vr30, vl30,vcm30,frame_R_heel_sground,frame_L_heel_sground,frame_R_heel_lground,frame_L_heel_lground =initial_read_data(IK_dir,trc_dir,dir_task,title)
    AUC_R,AUC_L,vertical_maxR,vertical_minR,vertical_maxL,vertical_minL =COM_analysis(cm,frame_R_heel_sground,dir_task,title)
    rms_final_steady,rms_start_end,rms_All,max_mean_velocity=Speed_analysis(vcm30,frame_R_heel_lground,frame_L_heel_lground ,dir_task,title)
    pace_r,temp_r,pace_l,temp_l=stride_length(data,frame_R_heel_sground,frame_L_heel_sground,dir_task,title)
    R_knee_steady,L_knee_steady,R_knee,L_knee=knee_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
    R_ankle_steady,L_ankle_steady,R_ankle,L_ankle=ankle_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
    R_hip_steady,L_hip_steady,R_hip,L_hip=hip_flexion_analysis(angle,dir_task,frame_R_heel_sground,frame_R_heel_lground,frame_L_heel_sground,frame_L_heel_lground,title)
    output = dict_output(R_hip_steady,L_hip_steady,R_knee_steady,L_knee_steady,R_ankle_steady,L_ankle_steady,max_mean_velocity,rms_final_steady,rms_start_end,rms_All,AUC_R,AUC_L,vertical_maxR,vertical_minR,vertical_maxL,vertical_minL,temp_r,temp_l)
    return output



####### Input parameter
IK_dir=r'./simulation/mocap_EMG_EEG_data/data_An_Yu/path1_02/opensim/Balancing_for_IK_BODY.mot'
trc_dir=r'./simulation/mocap_EMG_EEG_data/data_An_Yu/path1_02/opensim/Empty_project_filt_0-30.trc'
output_dir=r'./simulation/mocap_EMG_EEG_data/data_An_Yu/path1_02/preprocessing/output'
# output_dir=r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\Patient_ID\2024_05_07\2024_09_03_16_47_calculated\Walk1'
patient_id='Maurice_camtest2'
date_str='2024_06_24'
task_str='walk1'
# ######## Function to be called
output = gait1_singlefile(IK_dir,trc_dir,output_dir, patient_id, date_str, task_str)








#######Ignored Here

# dir_calculated = r'D:\Patient_data\Patient_ID\2024_05_07\2024_06_02_14_47_calculated'
# gait1(dir_calculated)