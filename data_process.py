import imp
import sqlite3
import numpy as np
import matplotlib.pyplot as plt


con = sqlite3.connect('output/sql_data.db')
cur = con.cursor()
table_name='_03_26_2022_19_43_27'

cur.execute("SELECT * FROM {}".format(table_name))

rows = cur.fetchall()

print(len(rows))
print(rows[0][1])

data = np.array(rows)

apply_time = data[2:,3] - data[1:-1,3]
fig1 = plt.figure()
plt.plot(apply_time)
plt.title('applying time interval')
print('average applying time interval: {}'.format(np.mean(apply_time)))



plt.show()