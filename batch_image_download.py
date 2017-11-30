import sys
import subprocess
import os
if len(sys.argv)<3:
    print("group_id output_dir start_id")
    quit()
sta=0
if sys.argv[3]!='':
    sta=int(sys.argv[3])
if os.path.isdir(sys.argv[2])==False:
    os.mkdir(sys.argv[2])
cmd="grep "+sys.argv[1]+" all.urls"
p=subprocess.Popen(cmd,shell=True,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
out,err=p.communicate()
urls=out.strip().split("\n")
n=0
for line in urls:
    if n>=sta:
        word=line.strip().split("\t")
        cmd1="wget "+word[-1]+" -O "+sys.argv[2]+"/"+str(n)+".jpg"
        p1=subprocess.Popen(cmd1,shell=True,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
        out1,err1=p1.communicate()
    n=n+1
    print(n)
