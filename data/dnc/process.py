

if __name__ == "__main__":
    readpath="dnc.csv"
    readf = open(readpath,encoding='utf-8')
    writepath="1dnc.csv"
    writef= open(writepath,"a+")
    lines=readf.readlines()
    for i,line in enumerate(lines):
        if i==0:
            writef.writelines(line)
            print(line)
        else:
            splits=line.split(',')
            u,v,t,T=splits
            u=int(u)-1
            v=int(v)-1
            context=str(u)+","+str(v)+","+str(t)+","+T
            print(context,end="")
            writef.writelines(context)
    readf.close()
    writef.close()

