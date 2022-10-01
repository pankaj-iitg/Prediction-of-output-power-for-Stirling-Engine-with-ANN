import numpy as np
import pandas as pd
import random 
import math

def log_sigmoid(x):
    return 1/(1+np.exp(-x))

eta= 0.85  # Learning rate
P = 112
M = 10   #int(input("Enter No. of hidden Neuron: "))
N = 1
L = 3
in_url = "in.xlsx"
df =pd.read_excel(in_url)
Ii = np.array(df)  
# print("Input \n",Ii)
Ii = Ii.astype(float).transpose()
# print("\ntransposed input :\n",Ii)

# normalization
for l in range (L):
    a = Ii[l].min()
    b = Ii[l].max()
    # print("a,b\n",a,b)
    for p in range (P):
        Ii[l][p] = (0.1+0.8*((Ii[l][p]-a)/(b-a)))
# print("\nNormalized input\n",Ii)

bias_value=[]
for p in range(P):
    bias_value.append(1)
# print("\nbias value:\n",bias_value)
I = np.vstack((bias_value,Ii))
I = I.transpose()
# print("\nfinal input I\n",I)

# v matrix
v = np.zeros(((L+1),M))
for l in range(L+1):
    for m in range(M):
        v[l][m] = (-1 + 2*random.random()) # min + (max-min)*random.random()
# print("weight v \n",v)

# # w matrix
w = np.zeros(((M+1),N))
for m in range(M+1):
    for n in range(N):
        w[m][n] = (-1 + 2*random.random()) # min + (max-min)*random.random()
# print("weight w \n",w)

# read output data
out_url = "output.xlsx"
df1 =pd.read_excel(out_url)
Od = np.array(df1)  
# print("desired output \n",Od)
Od = Od.transpose()
# print("desired output \n",Od)

# normalized output of desired output
for n in range (N):
    c = Od[n].min()
    d = Od[n].max()
    for p in range (P):
        Od[n][p] = (0.1+0.8*((Od[n][p]-c)/(d-c)))
Od=Od.transpose()
# print("\nnormalized desired output Od:\n", Od)

with open("iteration_mse.txt","w") as f1:
    f1.write("Iteration_no.\t\tmse")

with open("w_new.txt","w") as f3:
    f3.write("w_new\n")

with open("v_new.txt","w") as f4:
    f4.write("v_new\n")

# mse
iter = 0
mse = 5
while(mse>0.00015):
    # Ih = 0
    Ih = (np.dot(I,v))
    # print("Ih\n",Ih)
    # output of hidden layer
    oh = np.zeros((P,M))
    for p in range(0,P,1):
        for m in range(0,M,1):
            oh[p][m]=0
            oh[p][m] = log_sigmoid(Ih[p][m])
            Ih[p][m] = 0
    # print("oh\n",oh)
    hidden_bias = []
    for p in range (P):
        hidden_bias.append(1)
    # print(hidden_bias)
    # final output of hidden layer
    Oh = np.vstack((hidden_bias,oh.transpose())).transpose()
    # print("\nfinal output of hidden after TF, Oh\n",Oh)
    # input to output layer layer
    Io = 0
    Io = (np.dot(Oh,w))
    # print("\n input to output IO:\n",Io)
    # after TF
    Oo = np.zeros((P,N))
    for p in range(P):
        for n in range(N):
            Oo[p][n]=0
            Oo[p][n] = log_sigmoid(Io[p][n])
            Io[p][n] = 0
    # print("output of output layer after TF, Oo:\n",Oo)
    # finding delW 
    del_w = np.zeros((M+1,N))
    for m in range(M+1):
        for n in range(N):
            del_w[m][n] = 0
            for p in range(P):
                del_w[m][n] = del_w[m][n] + (eta/P)*(Od[p][n]-Oo[p][n])*Oo[p][n]*(1-Oo[p][n])*Oh[p][m]
    # print("del_w:\n",del_w)
    # w_new values
    w = np.add(w,del_w)
    # print("\nw_new\n",w)
    # del_v 
    del_v = np.zeros((L+1,M)) 
    for l in range(L+1):
        for m in range(M):
            del_v[l][m] = 0
            for p in range(P):
                for n in range(N):
                    del_v[l][m] = del_v[l][m] + (eta/(N*P))*(Od[p][n]-Oo[p][n])*Oo[p][n]*(1-((Oo[p][n])**2))*w[m+1][n]*oh[p][m]*(1-oh[p][m])*I[p][l]
    # print("del_v\n",del_v)
    v = np.add(v,del_v)
    # print("v_new:\n",v)
    # finding Error
    err = 0
    for p in range(P):
        for n in range(N):
            err = err + (0.5)*(pow((Od[p][n]-Oo[p][n]),2))

    mse = 0
    mse = float(err/P)
    print("Iteration no.:\n",iter)
    print("mse\n",mse)
    iter = iter+1
    with open("iteration_mse.txt","a") as f1:
        f1.write("\n")
        f1.write(str(iter))
        f1.write("\t\t\t\t")
        f1.write(str(mse))

print("\nCode has worked fine")
print("\nfinal_w:\n",w)
print("\nfinal_v:\n",v)
print("\nfinal_mse:\n",mse)
print("\nfinal_iteration:\n",iter)


with open("w_new.txt","a") as f3:
    f3.write("\nfinal_w:\n")
    f3.write(str(w))

with open("v_new.txt","a") as f4:
    f4.write("\nfinal_v\n")
    f4.write(str(v))


# Testing

Ih = (np.dot(I,v))
# print("Ih\n",Ih)
# output of hidden layer
oh = np.zeros((P,M))
for p in range(0,P,1):
    for m in range(0,M,1):
        oh[p][m]=0
        oh[p][m] = log_sigmoid(Ih[p][m])
        Ih[p][m] = 0
# print("oh\n",oh)
hidden_bias = []
for p in range (P):
    hidden_bias.append(1)
# print(hidden_bias)

# final output of hidden layer
Oh = np.vstack((hidden_bias,oh.transpose())).transpose()
# print("\nfinal output of hidden after TF, Oh\n",Oh)
# input to output layer layer
Io = 0
Io = (np.dot(Oh,w))
# print("\n input to output IO:\n",Io)

# after TF
Oo = np.zeros((P,N))
for p in range(P):
    for n in range(N):
        Oo[p][n]=0
        Oo[p][n] = log_sigmoid(Io[p][n])

with open("testing.txt","w") as f5:
    f5.write("act_data\t\t\t\t\tANN output\t\t\t\t\tError")

for p in range(100,P,1):
    for n in range(N):
        act = float(Od[p][n])
        ann = float(Oo[p][n])
        error = abs(act-ann)
        with open("testing.txt","a") as f5:
            f5.write("\n")
            f5.write(str(act))
            f5.write("\t\t\t")
            f5.write(str(ann))
            f5.write("\t\t\t")
            f5.write(str(error))



        




