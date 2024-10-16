import numpy as np
import cv2
import os

def read_mp4_files(folder_path):
    mp4files=[]
    filenames=[]
    total_frames=[]
   
    for file in os.listdir(folder_path):
        if file.endswith('.mp4'):
            # os.path.basename(file)
            mp4files.append(os.path.join(folder_path, file))
            filenames.append( os.path.basename(os.path.join(folder_path, file)))
            cap=cv2.VideoCapture(os.path.join(folder_path, file))
            if not cap.isOpened():
                print (f'cannot open {os.path.join(folder_path, file)}')
                total_frames.append(-1)
            else:    
                nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames.append(nframes)
                cap.release()

    return (filenames, mp4files, total_frames)

def read_txt_files(folder_path, suffix):
    filespath, filenames = [],[]
    bbxdata=[]
    for file in os.listdir(folder_path):
        if file.startswith(suffix) and file.endswith('.txt'):

            filespath.append(os.path.join(folder_path, file))
            filenames.append( file[:-4])
            with open(os.path.join(folder_path, file),'r') as bbx:
                temp1=bbx.read()[:-2]
                temp=(temp1.split())
                bbx_float=[float(item) for item in temp]
                bbxdata.append(bbx_float)

    
    return (filenames, filespath,bbxdata)

def frame_orders(txtfilelist):
    frame_names=[]
    for i in range(len(txtfilelist)):
        frame_names.append(int(txtfilelist[i].split('_')[-1]))
    return frame_names

def creat_missing_txt_orders(txtfilesample , idx):
    
    
    txtlist= txtfilesample.split('_')
    txtlist[-1]=str(idx)
    txtstr='_'.join(txtlist)
    return txtstr

def create_missing_indeces(frame_nums, bbx, sorted_frames ,txtfullpath_sorted):
    bbx_all=[]
    missing_idx=[]
    txtfile=[]
    txt_template=txtfullpath_sorted[0]
    frame_ids=np.zeros((frame_nums))#list(range(len(frame_nums)))
    for i in range(frame_nums):

        if i+1 in sorted_frames:
            bbx_all.append(bbx[i])
            txtfile.append(txtfullpath_sorted[i])
            frame_ids[i]=1
        else:
            bbx_all.append([])
            missing_idx.append(i+1)
            fname=creat_missing_txt_orders(txt_template , i+1)
            txtfile.append(fname)

    return (bbx_all, missing_idx, txtfile)

def count_lines(filename):
    with open(filename, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count



def read_txt_files_return_miss(folder_path, total_frame):
    missed_frames = []
    double_bbx_id=[]
    bbxdata=[]
    # frame0005.txt
    # print(f"total frame: {total_frame}")
    for  i in range(total_frame):#file in os.listdir(folder_path):
        frame_path=os.path.join(folder_path,f"frame{i:04d}.txt")
        # print(frame_path)
        # print('*******inside read txt ********')
        if os.path.exists(frame_path):#file.endswith('.txt'):
            # print('********file existed*******')
            with open(frame_path,'r') as bbx:
                if count_lines(frame_path)==1:
                    temp1=bbx.read()[:-2]
                    temp=(temp1.split())
                    bbx_float=[float(item) for item in temp]
                    bbxdata.append(bbx_float)
                else:
                    double_bbx_id.append(i)
                    bbxdata.append([])
                    
        else:
            # print(f'{frame_path}: does NOT exist')
            bbxdata.append([])
            missed_frames.append(i)
        ret_dat=(missed_frames,double_bbx_id, bbxdata)

    return ret_dat
def list_to_str(list_num):
    return ','.join(map(str,list_num))

def create_folders_and_file(path, number ,bbx_str):
    # Create folders if they don't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Format the number with four digits
    formatted_number = '{:04d}'.format(number)
    
    # Create the text file with the formatted number
    file_name = os.path.join(path, f"frame{formatted_number}.txt")
    with open(file_name, 'w') as file:
        file.write(bbx_str)
        print(f"{file_name}: with {bbx_str} created!")

def weighted_average(lst):
    # Calculate the middle index
    middle_index = len(lst) // 2
    
    # Calculate the weights using a kernel function
    weights = [1 / (abs(i - middle_index) + 1) for i in range(len(lst))]
    
    # Normalize the weights
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]
    
    # Calculate the weighted sum
    weighted_sum = sum(value * weight for value, weight in zip(lst, normalized_weights))
    
    # Round the weighted sum to the nearest integer
    # rounded_weighted_sum = round(weighted_sum)
    
    return weighted_sum 

def get_folders(directory , name):
    # Get all files and folders in the directory
    files_and_folders = os.listdir(directory)
    
    # Filter out only the folders
    folders = [item for item in files_and_folders if os.path.isdir(os.path.join(directory, item))]
    name_path=[os.path.join(directory, os.path.join(item, name)) for item in folders]
    return folders,name_path

def estimated_missing_bbxes(yolo_detections, missing_frames, wndw_size , miss_folder):
    # Estimated bounding box coordinates for missing frames
    # estimated_bboxes = []
    # print(f"{miss_folder}:{missing_frames}")
    w=wndw_size
    for missing_frame in missing_frames:
        st_id=max(0, missing_frame-(wndw_size//2))
        end_id=min(len(yolo_detections)-1, missing_frame+(wndw_size//2))
        xtopL = [sublist[1] for sublist in yolo_detections[st_id:end_id] if sublist]
        ytopL = [sublist[2] for sublist in yolo_detections[st_id:end_id] if sublist]
        xBotR = [sublist[3] for sublist in yolo_detections[st_id:end_id] if sublist]
        yBotR = [sublist[4] for sublist in yolo_detections[st_id:end_id] if sublist]
        cnf_all=[sublist[-1] for sublist in yolo_detections[st_id:end_id] if sublist]
        # check if lists are empty 
        if not xtopL:
            w=w+3
            estimated_missing_bbxes(yolo_detections, missing_frames, w,miss_folder)
        estimated_xtopl=round(weighted_average(xtopL))
        estimated_ytopl=round(weighted_average(ytopL))
        estimated_xbotr=round(weighted_average(xBotR))
        estimated_ybotr=round(weighted_average(yBotR))
        estimated_conf=round(weighted_average(cnf_all),2)
        class_id=0.0
        missed_bbx=[class_id, estimated_xtopl,estimated_ytopl,estimated_xbotr,estimated_ybotr ,estimated_conf]
        yolo_detections[missing_frame]= missed_bbx
        
        create_folders_and_file(miss_folder, missing_frame ,list_to_str(missed_bbx))
        
        

    return yolo_detections

