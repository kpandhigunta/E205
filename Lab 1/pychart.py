import numpy as np
import matplotlib.pyplot as plt

labels = ["Ages 0-15", "Ages 16-20", "Ages 21-25", "Ages 26-30", "Ages 30+"]
black = np.array([
13,
229,
423,
322,
938,
]) / 1925
white = np.array([
11,
136,
332,
490,
2386,
]) / 3355
hispanic = np.array([
12,
110,
225,
237,
713,
]) / 1297
asian = np.array([
0,
11,
10,
18,
79,
]) / 118



plt.figure()
plt.pie(black, labels = ['p('+labels[i]+'|black)'+'\n'+str(x)[:5] for i,x in enumerate(black)])
plt.savefig('black.png', dpi=300)

plt.figure()
plt.pie(white, labels = ['p('+labels[i]+'|white)'+'\n'+str(x)[:5] for i,x in enumerate(white)])
plt.savefig('white.png', dpi=300)

plt.figure()
plt.pie(hispanic, labels = ['p('+labels[i]+'|hispanic)'+'\n'+str(x)[:5] for i,x in enumerate(hispanic)])
plt.savefig('hispanic.png', dpi=300)

plt.figure()
plt.pie(asian, labels = ['p('+labels[i]+'|asian)'+'\n'+str(x)[:5] for i,x in enumerate(asian)])
plt.savefig('asian.png', dpi=300)
