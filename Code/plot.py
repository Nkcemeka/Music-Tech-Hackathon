import matplotlib.pyplot as plt

# Data
diffusion_steps = [100, 90, 80, 70, 60, 50]
alignment_cost = [0.0, 153406, 179681, 177482, 189131, 233754]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(diffusion_steps, alignment_cost, marker='o', linestyle='-', color='b')

# Adding title and labels
plt.title('Prompt: Jazz with Sax Solo 120bpm', fontsize=14)
plt.xlabel('Diffusion Steps', fontsize=12)
plt.ylabel('Alignment Cost', fontsize=12)

# Adding grid
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()
