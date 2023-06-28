import pandas
import matplotlib.pyplot as plt
import numpy as np

df = pandas.read_excel('rq3_avg.xlsx')
top1, top5, top10 = df[1], df[5], df[10]

labels = ['SUM','DESC','SUM+DESC','SUM*2+DESC','SUM*3+DESC','SUM*4+DESC','SUM+DESC*2','SUM+DESC*3','SUM+DESC*4']

x = np.array([1,2,3,4,5,6,7,8,9])
plt.bar(x-0.2,top1, width=0.2, label='Top-1 Acc')
plt.bar(x,top5,width=0.2, label='Top-5 Acc')
plt.bar(x+0.2,top10,width=0.2, label='Top-10 Acc')
plt.ylabel('Top-k Acc%')
plt.ylim(0,100)
plt.xticks(x, labels, rotation=90)
plt.subplots_adjust(top=0.95, bottom=0.3)
plt.legend()
plt.show()

