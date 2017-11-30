import sys
import subprocess
import os
import cv2
if len(sys.argv)<3:
    print("input_folder output_folder #num")
    quit()
if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])
cmd="ls "+sys.argv[1]
p=subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
out,err=p.communicate()
fls=out.rstrip("\r\n").split("\n")
n=0
for fl in fls: 
    pth=sys.argv[1]+"/"+fl
    img=cv2.imread(pth)
    try:
        ht,wt,ch=img.shape
        if ht>=200 and wt>=200 and ch==3:
            os.system("cp "+pth+" "+sys.argv[2]+"/"+str(n)+".jpg")
            n+=1
        if n>=int(sys.argv[3]):
            break
    except:
        continue

print n

