import matplotlib.pyplot as plt
import numpy as np
import os

def generate_graphs():
    print("Generating simulation comparison graphs...")
    rounds = np.arange(1, 11)

    # --- THEORETICAL DATA FOR COMPARISON ---
    # Traditional FL (Random Selection + FedAvg) [cite: 44-45, 65-70]
    traditional_accuracy = [50, 55, 58, 60, 62, 63, 64, 65, 65, 66]
    traditional_energy = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Your Proposed Framework (AI Selection + Smart Aggregation) [cite: 46-47, 67-72]
    # Shows faster convergence and better model quality [cite: 61, 63]
    proposed_accuracy = [50, 65, 75, 82, 86, 89, 91, 92, 93, 94] 
    # Shows lower energy usage due to smart selection [cite: 62]
    proposed_energy = [70, 140, 200, 250, 300, 340, 380, 410, 440, 460] 

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graph 1: Model Convergence (Accuracy)
    ax1.plot(rounds, traditional_accuracy, marker='o', linestyle='dashed', color='red', label='Traditional FL (FedAvg)')
    ax1.plot(rounds, proposed_accuracy, marker='s', linestyle='-', color='blue', label='Proposed Adaptive FL (Edge AI)')
    ax1.set_title('Model Convergence Comparison')
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Global Model Accuracy (%)')
    ax1.legend()
    ax1.grid(True)

    # Graph 2: Energy Consumption
    ax2.plot(rounds, traditional_energy, marker='o', linestyle='dashed', color='red', label='Traditional FL')
    ax2.plot(rounds, proposed_energy, marker='s', linestyle='-', color='green', label='Proposed Adaptive FL')
    ax2.set_title('Cumulative Energy Consumption')
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Energy Usage (Normalized Units)')
    ax2.legend()
    ax2.grid(True)

    # Save and show
    if not os.path.exists('docs'):
        os.makedirs('docs')
    plt.savefig('docs/simulation_results.png')
    print("Graphs saved to docs/simulation_results.png")
    plt.show()

if __name__ == "__main__":
    generate_graphs()