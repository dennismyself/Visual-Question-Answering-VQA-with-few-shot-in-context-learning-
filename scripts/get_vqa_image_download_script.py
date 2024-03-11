import os 
import argparse
#import cv2
#from PIL import Image
def main(args):
    key = args.image_key
    key_list = key.split()
    file_list = []
    for key in key_list:

        zero_pad = 12 - len(key)
        padded_key = zero_pad * "0" + key
        file_list.append("/path/to/shared/data/vqa2/val2014/COCO_val2014_"+padded_key+".jpg")
    files_str = ' '.join(file_list)
    # You may need to change this path if you did not work under this directory
    #path = "/home/"+ str(args.crsid)+"/rds/hpc-work/MLMI-VQA-2022/data/vqa2/val2014/COCO_val2014_"+padded_key+".jpg"
    command = "scp " + str(args.crsid) + "@"+ "login-gpu.hpc.cam.ac.uk:\" "+files_str+"\" ./"
    #scp remote:"A/1.txt A/2.txt B/1.txt C/1.txt" local:./
    # scp jm2245@login-gpu.hpc.cam.ac.uk:/home/jm2245/rds/hpc-work/MLMI-VQA-2022/data/vqa2/val2014/COCO_val2014_000000000042.jpg ./
    # scp jm2245@login-gpu.hpc.cam.ac.uk:/home/jm2245/rds/hpc-work/MLMI-VQA-2022/data/vqa2/val2014/COCO_val2014_000000000042.jpg /home/jm2245/rds/hpc-work/MLMI-VQA-2022/data/vqa2/val2014/COCO_val2014_COCO_val2014_000000000073.jpg ./
    # scp jm2245@login-gpu.hpc.cam.ac.uk:"/home/jm2245/rds/hpc-work/MLMI-VQA-2022/data/vqa2/val2014/COCO_val2014_000000000042.jpg /home/jm2245/rds/hpc-work/MLMI-VQA-2022/data/vqa2/val2014/COCO_val2014_COCO_val2014_000000000073.jpg" ./
    # scp jm2245@login-gpu.hpc.cam.ac.uk:/home/jm2245/rds/hpc-work/MLMI-VQA-2022/data/vqa2/val2014/\{COCO_val2014_000000000042.jpg,COCO_val2014_COCO_val2014_000000000073.jpg\} ./
    print(command)
    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--image_key",
        default="1 100 111",
        type=str,
        help="The image key in integer or string",
    )

    arg_parser.add_argument(
        "--crsid",
        default="abc123",
        help="Your crsid",
    )
    
    args = arg_parser.parse_args()
    main(args)