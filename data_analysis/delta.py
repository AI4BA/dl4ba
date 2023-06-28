from cliffs_delta import cliffs_delta
import pandas

df = pandas.read_excel('jdt_NextBug.xlsx')
top1, top5, top10 = df[1], df[5], df[10]
models = ['Bi-LSTM-A+NextBug','Bi-LSTM+NextBug','Bi-LSTM-A+ELMo','Bi-LSTM+ELMo','TextCNN+NextBug','MLP+NextBug','LSTM-A+NextBug','TextCNN+ELMo','LSTM-A+ELMo','LSTM+ELMo','LSTM+NextBug','MLP+ELMo']

def load_data(data):
    count = 9
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

print("Top1")
for i in range(len(top1)):
    for j in range(i+1,len(top1)):
        d, res = cliffs_delta(top1[i],top1[j])
        print(f"{models[i]} vs {models[j]}: {d:.4f}, {res}")

print("Top5")
for i in range(len(top5)):
    for j in range(i+1,len(top5)):
        d, res = cliffs_delta(top5[i],top5[j])
        print(f"{models[i]} vs {models[j]}: {d:.4f}, {res}")

print("Top10")
for i in range(len(top10)):
    for j in range(i+1,len(top10)):
        d, res = cliffs_delta(top10[i],top10[j])
        print(f"{models[i]} vs {models[j]}: {d:.4f}, {res}")
