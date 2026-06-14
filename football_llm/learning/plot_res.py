"""\cellcolor{mygray} DAAC       &	\cellcolor{mygray} \textbf{1.00$\pm$0.00} & 	\cellcolor{mygray}\textbf{0.94$\pm$0.03}  &  	\cellcolor{mygray} \textbf{0.81$\pm$0.02}        &  	\cellcolor{mygray}\textbf{0.76$\pm$0.02}  & 	\cellcolor{mygray}\textbf{0.92$\pm$0.02} & 	\cellcolor{mygray}\textbf{0.87$\pm$0.04} &   	\cellcolor{mygray}\textbf{0.86$\pm$0.02}  &   	\cellcolor{mygray}\textbf{0.77$\pm$0.03} & 	\cellcolor{mygray}\textbf{0.77$\pm$0.03}  & 	\cellcolor{mygray}\textbf{0.73$\pm$0.02}     \\
		& \cellcolor{mywhite} DCRL       &     0.99$\pm$0.01   & 0.93$\pm$0.01 & 0.78$\pm$0.03 &  0.74$\pm$0.03   & 0.44$\pm$0.03 & 0.32$\pm$0.02  & 0.31$\pm$0.00 & 0.51$\pm$0.01  &   0.50$\pm$0.02    &   0.46$\pm$0.02 \\
		& \cellcolor{mywhite} OSIL   &    0.43$\pm$0.09 & 0.16$\pm$0.10  & 0.14$\pm$0.10  &  0.04$\pm$0.02   & 0.50$\pm$0.07 & 0.29$\pm$0.05 & 0.30$\pm$0.07 & 0.32$\pm$0.05 &   0.22$\pm$0.03   &   0.21$\pm$0.04 \\
		& \cellcolor{mywhite} CbMRL      &       0.98$\pm$0.00    & 0.76$\pm$0.02 & 0.66$\pm$0.01 &  0.44$\pm$0.02  & 0.28$\pm$0.02 & 0.29$\pm$0.03 & 0.26$\pm$0.03 & 0.37$\pm$0.03  &   0.32$\pm$0.03   &  0.33$\pm$0.02 \\

		% \cmidrule{2-12} 
		% & \cellcolor{mywhite} DCRL-final &    0.96       &  0.92         & 0.76          &  0.72         &      0.45     & 0.45  &     0.42      &   0.44      &       0.43   & 0.46 \\
  
  		\midrule

		\multirow{4}{*}{\vspace*{\fill} \rotatebox[origin=c]{90}{No-Coord}} &	\cellcolor{mygray} DAAC       &	\cellcolor{mygray} \textbf{0.51$\pm$0.19} & 	\cellcolor{mygray}\textbf{0.71$\pm$0.06}  &   	\cellcolor{mygray}\textbf{0.46$\pm$0.06}        &  	\cellcolor{mygray} \textbf{0.58$\pm$0.04}          & 	\cellcolor{mygray} \textbf{0.83$\pm$0.03} &	\cellcolor{mygray} \textbf{0.63$\pm$0.01} & 	\cellcolor{mygray}  \textbf{0.54$\pm$0.05}  &  	\cellcolor{mygray} \textbf{0.50$\pm$0.02}     & 	\cellcolor{mygray} \textbf{0.45$\pm$0.02}    & 	\cellcolor{mygray} \textbf{0.40$\pm$0.03}     \\
		&  \cellcolor{mywhite} DCRL       &     0.24$\pm$0.03  & 0.01$\pm$0.01 & 0.15$\pm$0.01 &  0.00$\pm$0.00  & 0.15$\pm$0.06 & 0.05$\pm$0.02  & 0.04$\pm$0.02 & 0.11$\pm$0.02  &   0.03$\pm$0.02    &  0.05$\pm$0.02 \\
		&\cellcolor{mywhite} OSIL   &    0.06$\pm$0.02   & 0.00$\pm$0.00 & 0.02$\pm$0.02 &  0.00$\pm$0.00  & 0.06$\pm$0.04 & 0.01$\pm$0.01  & 0.02$\pm$0.01 & 0.02$\pm$0.03 &  0.03$\pm$0.02  &  0.01$\pm$0.01 \\
		& \cellcolor{mywhite} CbMRL      &      0.15$\pm$0.06    & 0.02$\pm$0.01 & 0.09$\pm$0.02 &  0.01$\pm$0.00  & 0.14$\pm$0.01 & 0.03$\pm$0.01  & 0.02$\pm$0.01 & 0.10$\pm$0.02  &   0.05$\pm$0.02    &   0.06$\pm$0.01 \\"""





"""
User
Extract numeric mean values from a LaTeX table. Each row is an algorithm, each column is a task, and each cell contains a mean and variance. Ignore LaTeX formatting commands and return only the floating-point means. Table:\cellcolor{mygray} DAAC       &	\cellcolor{mygray} \textbf{1.00$\pm$0.00} & 	\cellcolor{mygray}\textbf{0.94$\pm$0.03}  &  	\cellcolor{mygray} \textbf{0.81$\pm$0.02}        &  	\cellcolor{mygray}\textbf{0.76$\pm$0.02}  & 	\cellcolor{mygray}\textbf{0.92$\pm$0.02} & 	\cellcolor{mygray}\textbf{0.87$\pm$0.04} &   	\cellcolor{mygray}\textbf{0.86$\pm$0.02}  &   	\cellcolor{mygray}\textbf{0.77$\pm$0.03} & 	\cellcolor{mygray}\textbf{0.77$\pm$0.03}  & 	\cellcolor{mygray}\textbf{0.73$\pm$0.02}     \\
		& \cellcolor{mywhite} DCRL       &     0.99$\pm$0.01   & 0.93$\pm$0.01 & 0.78$\pm$0.03 &  0.74$\pm$0.03   & 0.44$\pm$0.03 & 0.32$\pm$0.02  & 0.31$\pm$0.00 & 0.51$\pm$0.01  &   0.50$\pm$0.02    &   0.46$\pm$0.02 \\
		& \cellcolor{mywhite} OSIL   &    0.43$\pm$0.09 & 0.16$\pm$0.10  & 0.14$\pm$0.10  &  0.04$\pm$0.02   & 0.50$\pm$0.07 & 0.29$\pm$0.05 & 0.30$\pm$0.07 & 0.32$\pm$0.05 &   0.22$\pm$0.03   &   0.21$\pm$0.04 \\
		& \cellcolor{mywhite} CbMRL      &       0.98$\pm$0.00    & 0.76$\pm$0.02 & 0.66$\pm$0.01 &  0.44$\pm$0.02  & 0.28$\pm$0.02 & 0.29$\pm$0.03 & 0.26$\pm$0.03 & 0.37$\pm$0.03  &   0.32$\pm$0.03   &  0.33$\pm$0.02 \\"""

# Results after switching multi-map non-observation and observation settings.
# DDT: [1.00, 0.94, 0.81, 0.76, 0.92, 0.87, 0.86, 0.77, 0.77, 0.73]
# DCRL: [0.99, 0.93, 0.78, 0.74, 0.51, 0.50, 0.46, 0.44, 0.32, 0.31]
# OSIL: [0.43, 0.16, 0.14, 0.04, 0.50, 0.29, 0.30, 0.32, 0.22, 0.21]
# CbMRL: [0.98, 0.76, 0.66, 0.44, 0.37, 0.32, 0.33, 0.28, 0.29, 0.26]
import numpy as np

data=np.array([
    [1.00, 0.94, 0.81, 0.76, 0.92, 0.87, 0.86, 0.77, 0.77, 0.73],
    [0.99, 0.93, 0.78, 0.74, 0.51, 0.50, 0.46, 0.44, 0.32, 0.31],
    [0.98, 0.76, 0.66, 0.44, 0.37, 0.32, 0.33, 0.28, 0.29, 0.26],
	[0.43, 0.16, 0.14, 0.04, 0.50, 0.29, 0.30, 0.32, 0.22, 0.21],
])
data_std = np.array([
    [0.0, 0.03, 0.02, 0.02, 0.02, 0.04, 0.02, 0.03, 0.03, 0.02],
    [0.01, 0.01, 0.03, 0.03, 0.03, 0.02, 0.0, 0.01, 0.02, 0.02],
    [0.09, 0.1, 0.1, 0.02, 0.07, 0.05, 0.07, 0.05, 0.03, 0.04],
    [0.0, 0.02, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.02]
])
key_to_col_indx={
    'sm_nonobs_seen':0,
    'sm_nonobs_new_demo':1,
	'sm_obs_seen':2,
    'sm_obs_new_demo':3,

	'2500_nonobs_seen':4,
    '2500_nonobs_new_demo':5,
    '2500_nonobs_new_map':6,
    '2500_obs_seen':7,
    '2500_obs_new_demo':8,
    '2500_obs_new_map':9,
}


figs_keys={
	'Train':  lambda x: x in ['sm_nonobs_seen', 'sm_obs_seen', '2500_nonobs_seen', '2500_obs_seen'],  #lambda x: '_seen' in x,
    'Non-Obstacle': lambda x: x in ['sm_nonobs_new_demo', '2500_nonobs_new_demo', '2500_nonobs_new_map'],    # lambda x: '_nonobs_' in x and '_new_' in x,
    'Unforseen Obstacle': lambda x: x in  ['sm_obs_new_demo', '2500_obs_new_demo', '2500_obs_new_map'],  # lambda x: '_obs_' in x  and '_new_' in x,
    # 'Single Map': lambda x: x in ['sm_nonobs_new_demo', 'sm_obs_new_demo'],
    # 'Multi-Map': lambda x: x in ['2500_nonobs_new_demo', '2500_obs_new_demo', '2500_nonobs_new_map', '2500_obs_new_map'],
}

key_of_train = ['sm_nonobs_seen', 'sm_obs_seen', '2500_nonobs_seen', '2500_obs_seen']
plot_curve_keys = lambda x: x in key_of_train

figs_results=np.zeros(shape=(4, len(figs_keys.keys())))

figs_std = np.zeros(shape=(4, len(figs_keys.keys())))

plot_curve_results = np.zeros(shape=(4, len(key_of_train)))
plot_curve_std = np.zeros(shape=(4, len(key_of_train)))



for fig_name in figs_keys.keys():
    col_indx = []
    col_indx = [key_to_col_indx[key] for key in key_to_col_indx.keys() if figs_keys[fig_name](key)]
    figs_results[:, list(figs_keys.keys()).index(fig_name)] = np.mean(data[:, col_indx], axis=1)
    figs_std[:, list(figs_keys.keys()).index(fig_name)] = np.std(data[:, col_indx], axis=1)



import seaborn as sns
sns.set_style('darkgrid', {'legend.frameon': True})
import matplotlib.pyplot as plt
algorithms = ["DDT", "DCRL", "CbMRL", "Trans4OSIL"]



means = np.array(figs_results)

# Number of tasks (columns) and algorithms (rows)
n_tasks = len(figs_keys.keys())
n_algorithms = len(algorithms)

# Width of a bar
bar_width = 0.2
PRETTY_COLORS = ['orangered',  'royalblue', 'forestgreen',  'orange', 'deeppink', 'deepskyblue']
PRETTY_MARKERS = ['o', 'v', 'd',  'P', 'p', ]
# Setting the positions of the bars
indices = np.arange(n_tasks)
positions = [indices + i*bar_width for i in range(n_algorithms)]

# Creating the bar plot
plt.figure(figsize=(12, 6))
for i, (algorithm, pos) in enumerate(zip(algorithms, positions)):
    plt.bar(pos, means[i], width=bar_width, label=algorithm, color=PRETTY_COLORS[i])
    # add error bar
    plt.errorbar(pos, means[i], yerr=figs_std[i], ecolor='black', capsize=5, fmt='none')
    # add text to bar
    for x, y in zip(pos, means[i]):
        plt.text(x, y+0.04, '%.2f' % y, ha='center', va='bottom', fontsize=16, color=PRETTY_COLORS[i], fontweight='bold')

# Adding labels and title
plt.ylabel('Success Rate', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(indices + bar_width, figs_keys.keys(), fontsize=20)

# set legend to the top of the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=20)


# Display the plot
plt.tight_layout()
plt.show()

# Save the plot
plt.savefig('performance.pdf')
plt.savefig('performance.png')
print(figs_results)
print((figs_results[:, 2] - figs_results[:, 1])/figs_results[:, 1])

col_indx = [key_to_col_indx[key] for key in key_to_col_indx.keys() if plot_curve_keys(key)]
plot_curve_results =data[:, col_indx]
plot_curve_std = data_std[:, col_indx]
plt.cla()

plt.figure(figsize=(12, 6))
for i in range(len(algorithms)):
    plt.plot(plot_curve_results[i], '--', label=algorithms[i], color=PRETTY_COLORS[i], marker=PRETTY_MARKERS[i], markersize=10)
    # plot error
    plt.fill_between(np.arange(len(plot_curve_results[i])), plot_curve_results[i]-plot_curve_std[i], 
                     plot_curve_results[i]+plot_curve_std[i], alpha=0.2, color=PRETTY_COLORS[i])


# add xtick
obj = []
obj.extend(plt.yticks(fontsize=20)[1])
obj.extend(plt.xticks(ticks=[0,1,2,3], labels=['Single-Map-NonObs', 'Single-Map-Obs', 'Multi-Map-NonObs', 'Multi-Map-Obs'], fontsize=18)[1])
# plt.xlabel('Tasks', fontsize=19)
obj.append(plt.xlabel('Tasks', fontsize=19))
obj.append(plt.ylabel('Success Rate', fontsize=19))
# plt.ylabel('Success Rate', fontsize=19)
obj.append(plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=20))
plt.tight_layout()
plt.savefig('perf-curve.png', bbox_extra_artists=obj, bbox_inches='tight')

plt.savefig('perf-curve.pdf', bbox_extra_artists=obj, bbox_inches='tight')
