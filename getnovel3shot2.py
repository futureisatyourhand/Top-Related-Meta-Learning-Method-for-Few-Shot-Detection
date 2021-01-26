import sys
filename=sys.argv[1]
f=open(filename,'r')
content=f.readlines()
f.close()
dic=[]
for c in content:
    if "comp4_det_test_" in c:
        print(c.split('/'))
        if int(c.split('/')[-2][-2])==1:
            dic.append([int(c.split('/')[-2][-2:])])
        elif int(c.split('/')[-2][-2])==0:
            dic.append([int(c.split('/')[-2][-1])])
        elif int(c.split('/')[-2][-2])==2:
            dic.append([int(c.split('/')[-2][-2:])])
    if "Mean AP" in c and len(dic[-1])==1:
        dic[-1].append(float(c.split('=')[-1][:-5]))
print(sorted(dic))

