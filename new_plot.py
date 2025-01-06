import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


y_values = [0.4752,0.4460,0.4206,0.3478]

# Corresponding x values
x_values = [0.3, 0.4, 0.5, 0.6]

def format_yticks(x, pos):
    return f"{x:.1f}"

# Plotting
plt.figure(figsize=(6, 4.5))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='orange', label="ROUGE R1")  # Dark blue (Teal)
#plt.plot(x_values, y1_values, marker='o', linestyle='-', color='darkseagreen', label="ROUGE R2")  # Coral
#plt.plot(x_values, y2_values, marker='o', linestyle='-', color='darkkhaki', label="ROUGE RL")

#plt.xlabel("X values", fontsize=12)
#plt.ylabel("Y values", fontsize=12)
#plt.title("Plot of Y values vs X values", fontsize=14)
# Set x-ticks explicitly
plt.xticks([0.3, 0.4, 0.5, 0.6])
plt.xlabel("Sparsity Level", fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.grid(alpha=0.5)
font1 ={'family': "Timew New Roman", 'weight':'normal', 'size': '12'}
#plt.legend(loc='upper right', prop=font1)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.yticks([0.35, 0.4, 0.45, 0.5])
plt.show()