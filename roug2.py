def calculate(one,two,three,four):
    a=[one,two,three,four]
    print(a,"a")
    b=[]
    c=[]
    d=[]
    for i in a:
        if i not in b:
            b.append(i)
        else:
            i=i+1
            for j,k in enumerate( b):
                if i==k:
                    #print(k)
                    n=k+1
                    c.append(n)
                    d.append(j)
            #print(d,c)
            for o ,j in enumerate (b):
                if len(c)>0:
                    if c[-1]==j:
                        m=j+1
                        b[o]=m
            if len(c)>0 and len(d)>0:
                b[d[-1]]=c[-1]
            b.append(i)
    ##        print(c,d)
    print("input",a)
    #print("output",b)
    #print(b)
    f=[]
    for i in b:
        f.append(i)
    f.sort()
    #print(b)
    #print(f)
    for i ,j in enumerate (b):
        if f[0]==j:
            b[i]="a"
        elif f[1]==j:
            b[i]="b"
        elif f[2]==j:
            b[i]="c"
        elif f[3]==j:
            b[i]="d"
    print("final output" ,b)
    return(b)

        
