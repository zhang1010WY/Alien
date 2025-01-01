loss_sup = []
loss_ce = []
f1_s = []
f1_t = []
num = 0
with open("nohup.out") as f:
    n = f.readlines()

    l1 = 466
    l2 = 466
    l3 = 483
    l4 = 489
    while(num<50):
        loss_sup.append(float(n[l1].strip().split(",")[0].split(" ")[2]))
        loss_ce.append(float(n[l2].strip().split(",")[1].split(" ")[3]))
        f1_s.append(float(n[l3].strip().split(",")[2].strip().split(":")[1]))
        f1_t.append(float(n[l4].strip().split(",")[2].strip().split(":")[1]))

        num += 1
        l1 += (910-466)
        l2 += (910-466)
        l3 += (910-466)
        l4 += (910-466)

    # print(n[466].strip().split(",")[0].split(" ")[2])
    # print(n[466].strip().split(",")[1].split(" ")[3])

    # print(n[483].strip().split(",")[2].strip().split(":")[1])
    # print(n[489].strip().split(",")[2].strip().split(":")[1])
print(loss_sup) 
print(loss_ce)
print(f1_s) 
print(f1_t)   

loss = []
for i in range(num):
    loss.append(loss_sup[i] + loss_ce[i])
print(loss)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,6))

x_data = range(1,num+1)
y_data1 = f1_t
y_data2 = f1_s

plt.plot(x_data,y_data1,color='red',linewidth=2.0,linestyle='--')
# plt.plot(x_data,y_data2,color='blue',linewidth=3.0,linestyle='-.')
plt.savefig('loss.png', bbox_inches='tight')
