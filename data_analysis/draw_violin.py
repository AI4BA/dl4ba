import pandas
import matplotlib.pyplot as plt

df = pandas.read_excel('firefox.xlsx')
top1, top5, top10 = df[1], df[5], df[10]

def load_data(data):
    count = 13
    start = 0
    end = count
    a = []
    for i in range(12):
        a.append(data[start:end])
        start = end
        end += count
    return a

top1 = load_data(top1)
top5 = load_data(top5)
top10 = load_data(top10)
x = ['TextCNN+ELMo','TextCNN+NextBug','LSTM+ELMo','LSTM+NextBug','Bi-LSTM+ELMo','Bi-LSTM+NextBug','LSTM-A+ELMo','LSTM-A+NextBug','Bi-LSTM-A+ELMo','Bi-LSTM-A+NextBug','MLP+ELMo','MLP+NextBug']

def draw(data):
    for i,each in enumerate(data):
        plt.violinplot(each,showmeans=True,positions=[i+1])

draw(top1)
a = [each for each in range(1,13)]
plt.xticks(a, x, rotation=90)
plt.ylabel('Top1 Acc%')
plt.ylim(0,100)
plt.subplots_adjust(top=0.95, bottom=0.3)
plt.show()

draw(top5)
plt.xticks(a, x, rotation=90)
plt.ylabel('Top5 Acc%')
plt.ylim(0,100)
plt.subplots_adjust(top=0.95, bottom=0.3)
plt.show()

draw(top10)
plt.xticks(a, x, rotation=90)
plt.ylabel('Top10 Acc%')
plt.ylim(0,100)
plt.subplots_adjust(top=0.95, bottom=0.3)
plt.show()


