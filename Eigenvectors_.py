from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np 
import scipy
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import seaborn as sns


f=open('data_sheets/Folds5x2_pp_binary_conversed.csv')
data=f.read()
f.close()
lines=data.split('\n')
header=lines[0].split(',')
lines=lines[1:9569]
data_x=np.zeros(((len(lines)),len(header)))
for i, line in enumerate(lines):
	values=[float(x) for x in line.split(',')[0:]]
	data_x[i,:]=values

ambTemp_reading=data_x[:,0]
vacuum_reading=data_x[:,2]
ambPress_reading=data_x[:,4]
relHumid_reading=data_x[:,6]

PE_reading=data_x[:,8]

ambTemp_int=data_x[:,1]
vacuum_int=data_x[:,3]
ambPress_int=data_x[:,5]
relHumid_int=data_x[:,7]

x=PE_reading
y=ambTemp_int


g = (sns.jointplot(x, y, kind="hex").set_axis_labels("Power Generated HMI Reading [MW]", "Ambient Temperature from PLC [Decimal]"))
g.annotate(stats.pearsonr)

plt.show()


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

nstd = 3
ax = plt.subplot(111)

cov = np.cov(x, y)
vals, vecs = eigsorted(cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
w, h = 2 * nstd * np.sqrt(vals)
ell = Ellipse(xy=(np.mean(x), np.mean(y)),
              width=w, height=h,
              angle=theta, color='black')
ell.set_facecolor('none')
ax.add_artist(ell)
plt.scatter(x, y)
plt.title("Ellipse of Interest .95")
plt.xlabel('Power Generated')
plt.ylabel('Ambient Temperature')
plt.show()

ambTemp_r_mean=np.mean(data_x[:,0])
vacuum_r_mean=np.mean(data_x[:,2])
ambPress_r_mean=np.mean(data_x[:,4])
relHumid_r_mean=np.mean(data_x[:,6])

PE_r_mean=np.mean(data_x[:,8])

ambTemp_int_mean=np.mean(data_x[:,1])
vacuum_int_mean=np.mean(data_x[:,3])
ambPress_int_mean=np.mean(data_x[:,5])
relHumid_int_mean=np.mean(data_x[:,7])

ambTemp_r_variance=np.var(data_x[:,0])
vacuum_r_variance=np.var(data_x[:,2])
ambPress_r_variance=np.var(data_x[:,4])
relHumid_r_variance=np.var(data_x[:,6])

PE_r_variance=np.var(data_x[:,8])

ambTemp_int_variance=np.var(data_x[:,1])
vacuum_int_variance=np.var(data_x[:,3])
ambPress_int_variance=np.var(data_x[:,5])
relHumid_int_variance=np.var(data_x[:,7])


PE_r_std=np.std(data_x[:,8])
ambTemp_int_std=np.std(data_x[:,1])
vacuum_int_std=np.std(data_x[:,3])
ambPress_int_std=np.std(data_x[:,5])
relHumid_int_std=np.std(data_x[:,7])


print ("Ambient Temperature Variance" , ambTemp_int_variance)
print ("Vacuum Variance" , vacuum_int_variance)
print ("Ambient Pressure Variance" , ambPress_int_variance)
print ("Relative Humidity Variance" , relHumid_int_variance)
print("Power Generated Variance",PE_r_variance)

print ("Ambient Temperature Mean" , ambTemp_r_mean)
print ("Vacuum Mean" , ambTemp_r_mean)
print ("Ambient Pressure Mean" , ambTemp_r_mean)
print ("Relative Humidity Mean" , ambTemp_r_mean)

print("Power Generated std",PE_r_std)
print("Ambient PRessure std",ambTemp_int_std)
print("Vamcuum std",vacuum_int_std)
print("Ambient std",ambPress_int_std)
print("Relative std",relHumid_int_std)


cx=np.cov(ambTemp_int=data_x[:,1],PE_reading=data_x[:,8])

print(cx)



