import csv
import matplotlib.pyplot as plt
file = open('reslut.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
for row in csvreader:
    if len(row)>0:
        rows.append(row)
XMX=[]
Throughput=[]
Latency=[]
Application_runtime=[]
FULL_GC=[]
GC=[]
RandomForest=[]
KNeighbors=[]
DecisionTree=[]
MLP=[]
Ensemble=[]
for i in range(0,len(rows)):
    if int(rows[i][0])%200==0 or int(rows[i][0])==2:
        XMX.append(rows[i][0])
        Throughput.append(float(rows[i][1]))
        Latency.append(float(rows[i][2]))
        Application_runtime.append(float(rows[i][3]))
        FULL_GC.append(float(rows[i][4]))
        GC.append(float(rows[i][5]))
        RandomForest.append(float(rows[i][6]))
        KNeighbors.append(float(rows[i][7]))
        DecisionTree.append(float(rows[i][8]))
        MLP.append(float(rows[i][9]))
        Ensemble.append(float(rows[i][10]))
plt.xticks(rotation=90)
plt.scatter(XMX, RandomForest, label= header[6], color= "green",marker="8", s=30)
plt.scatter(XMX, KNeighbors, label=header[7], color= "red",marker= "s", s=30)
plt.scatter(XMX, DecisionTree, label=header[8], color= "yellow",marker= "p", s=30)
plt.scatter(XMX, MLP, label=header[9], color= "blue",marker= "P", s=30)
plt.scatter(XMX, Ensemble, label=header[10], color= "black",marker= "*", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Accuracy(0%-100%)')
plt.title('Classifier Algorithms')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, RandomForest, label= header[6], color= "green",marker="8", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Accuracy(0%-100%)')
plt.title('Classifier Algorithm')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, KNeighbors, label=header[7], color= "red",marker= "s", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Accuracy(0%-100%)')
plt.title('Classifier Algorithm')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, DecisionTree, label=header[8], color= "yellow",marker= "p", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Accuracy(0%-100%)')
plt.title('Classifier Algorithm')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, MLP, label=header[9], color= "blue",marker= "P", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Accuracy(0%-100%)')
plt.title('Classifier Algorithm')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, Ensemble, label=header[10], color= "black",marker= "*", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Accuracy(0%-100%)')
plt.title('Classifier Algorithm')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, Latency, label=header[2], color= "green",marker="8", s=30)
plt.scatter(XMX, Application_runtime, label=header[3], color= "red",marker= "s", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Time(s)')
plt.title('Application Runtime/Latency Graph')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, Application_runtime, label=header[3], color= "red",marker= "s", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Time(s)')
plt.title('Application Runtime Graph')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, Throughput, label=header[1], color= "black",marker= "*", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Throughput(0%-100%)')
plt.title('Performance Graph')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, Latency, label=header[2], color= "green",marker="8", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('Time(s)')
plt.title('Performance Graph')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX,  FULL_GC, label=header[4], color= "blue",marker= "P", s=30)
plt.scatter(XMX, GC, label=header[5], color= "black",marker= "*", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('GC/FULL-GC Count')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, GC, label=header[5], color= "blue",marker= "P", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('GC Count')
plt.legend()
plt.show()
plt.xticks(rotation=90)
plt.scatter(XMX, FULL_GC, label=header[4], color= "yellow",marker= "p", s=30)
plt.xlabel('Xmx (SerialGC)')
plt.ylabel('FULL-GC Count')
plt.legend()
plt.show()
