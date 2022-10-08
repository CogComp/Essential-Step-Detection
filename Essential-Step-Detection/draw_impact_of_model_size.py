import matplotlib.pyplot as plt



NSP_parameter_size = [110, 350]
NSP_performance = [56.31, 56.31]

PPL_parameter_size = [117, 345, 774]
PPL_performance = [56.76, 59.27]


Prompt_parameter_size = [350, 1300, 6700, 175000]
Prompt_performance = [55.12, 53.18, 52.05, 63.59]

plt.xscale('log')
plt.plot(Prompt_parameter_size, Prompt_performance)
plt.plot(NSP_parameter_size, NSP_performance)
plt.show()
