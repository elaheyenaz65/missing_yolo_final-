import numpy as np
import os 
from helpers import *
import argparse

def main(root_dir, subject_id,save_path ):
    # subject_id='D16'
    subj_estimated_miss_folder=os.path.join(save_path ,subject_id)
    # root_dir='/media/eli/B95FCD0F6C3BB0A7/career_data/D16_data/'
    folders , video_path= get_folders(root_dir ,'video')
    folders , bbx_path= get_folders(root_dir ,'bbox')
    # print(folders)
    # print('==============folders================')
    assert(len(bbx_path)==len(video_path))
    for action in range(len(video_path)):
        fold_path=video_path[action]#'/media/eli/B95FCD0F6C3BB0A7/career_data/D16_data/crawling/video/'
        lbl_path=bbx_path[action]#'/media/eli/B95FCD0F6C3BB0A7/career_data/D16_data/crawling/bbox/'
        miss_frames_folders=os.path.join(subj_estimated_miss_folder, folders[action])
        flist, mp4list,total_frames=read_mp4_files(fold_path)
        # print(total_frames)
        window_size=5
        cntr=0
        thresh=3
        with open(subject_id+'_'+folders[action]+'_curroptedfile.txt', 'a') as f_currp, open(subject_id+'_'+folders[action]+'_double_bbx.txt', 'a') as f_two_bbx:
            for i in range (len(flist)):
                if total_frames[i]>0 :
                    crl=flist[i][:-4]
                    
                    lbl_path_bbx=os.path.join(lbl_path,crl)

                
                    missed_frames,double_bbx_id, bbxdata=read_txt_files_return_miss(lbl_path_bbx, total_frames[i])
                    # print(f"missed frame: {len(missed_frames)} frame with double bbxes:{len(double_bbx_id)} total_frame:{total_frames[i]}")
                    if  len(missed_frames)+len(double_bbx_id) >0:
                        cntr+=1
                        if len(missed_frames)> (total_frames[i]//thresh):
                            # print(f"file:{crl} missed frame: {len(missed_frames)} frame with double bbxes:{len(double_bbx_id)} total_frame:{total_frames[i]}")
                            f_currp.write(mp4list[i]+'\n')
                        else:
                            
                            final_bbxes=estimated_missing_bbxes(bbxdata,missed_frames,window_size,os.path.join(miss_frames_folders ,crl ))
                            # print(missed_frames)
                            # print('we need to work on that')
                            # input('here is missed frames, press any key to continue')
                            if len(double_bbx_id):
                                # print(mp4list[i])
                                # print(f'{double_bbx_id[0]}: {bbxdata[double_bbx_id[0]]}')
                                f_two_bbx.write(mp4list[i]+'\n')
                                f_two_bbx.write('double box frame numbers: ')
                                f_two_bbx.write(list_to_str(double_bbx_id))
                                f_two_bbx.write('\n')
                                # input('press any key to continue')
                else:
                    f_currp.write('cannot open ' + flist[i]+'\n')

                
                # print(cntr)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='this code estimates the missing frames bounding box')
    parser.add_argument('--root_dir', type=str, default='/media/eli/B95FCD0F6C3BB0A7/career_data/D16_data/', help='path to subject data')
    parser.add_argument('--subject', type=str, default=0, help='subject id eg:D16')
    parser.add_argument('--save', type=str, default='/home/eli/', help='save path for missing frames')

    args = parser.parse_args()

    main(args.root_dir, args.subject, args.save) 
