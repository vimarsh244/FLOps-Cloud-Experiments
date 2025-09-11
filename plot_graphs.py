import re
import matplotlib.pyplot as plt
import numpy as np

# Function to extract loss per round from log file
def extract_loss(file_path):
    rounds = []
    losses = []
    pattern = re.compile(r"round (\d+): ([\d\.]+)")

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                rounds.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    return rounds, losses

# Function to extract accuracy per round from log file
def extract_accuracy(file_path):
    rounds = []
    accuracies = []
    pattern = re.compile(r"\( *(\d+), ([\d\.]+)\)")  # Pattern for accuracy data in (round, value) format

    with open(file_path, "r") as file:
        in_metrics_section = False
        for line in file:
            if "History (metrics, distributed, evaluate):" in line:
                in_metrics_section = True
                continue
            if in_metrics_section:
                match = pattern.findall(line)
                for round_num, acc in match:
                    rounds.append(int(round_num))
                    accuracies.append(float(acc))

    return rounds, accuracies

# File paths
logs = {
    # "IID Baseline": "baseline.log",
    "Non-IID Baseline": "niid_baseline.log",
    # "Node Disconnect": "nodeDisconnect.log",
    #"Timeout": "timeout40s.log",
 #   "Non-IID Disconnect": "noniid_s42.log",
    # "Client 4 Disconnect": "client4_drop_redo.log",
    # "Representative distribution strategy": "dropped_client_substitution.log",
    # "Client 4 Disconnect with DIWS":   "dcs_lower_weightage.log",
    #"Central substitution": "client4_central_substitution.log",
    "Client 5 Disconnect": "client5_drop_baseline.log",
    #"Client 1 Disconenct Baseline": "client1_drop.log",
    # "Client 2 Disconnect": "client2_drop.log",
    #"Client 3 Disconenct Baseline": "client3_drop.log",
    # "Client 5 Disconnect with DIWS": "client5_substitution.log",
    #"Client 1 Substitution": "client1_drop_substitution.log",
    # "Client 2 Disconnect with DIWS": "client2_drop_substitution.log",
    #"Client 3 Substitution": "client3_drop_substitution.log",
    # "Client 2 & 5 Disconnect": "clients_2_5_drop.log",
    # "Client 2 & 5 Disconnect with DIWS": "clients_2_5_drop_substitution.log",
    # "IID Client 5 Drop": "iid_client5_drop.log",
    # "Client 2 & 5 IID Node Disconnect": "iid_clients_2_5_drop.log",
    # "Client 2 & 5 IID Substitution": "iid_clients_2_5_substitution.log",
    # "Multi Epochs": "multi_epoch.log",
    # "Non-IID MMProx Client 3 Disconnect": "niid_mmprox_client_3_drop.log",
    "Client 5 Disconnect with FedAdagrad": "client5_drop_fedadagrad_formatted.log"
    
}

# Colors and markers for plotting
plot_styles = {
    # "IID Baseline": ("blue", "o"),
    "Non-IID Baseline": ("blue", "o"),
    # "Node Disconnect": ("red", "s"),
    # "Timeout": ("green", "D"),
 #   "Non-IID Disconnect": ("orange", "^"),
    # "Client 4 Disconnect": ("red", "s"),
    #"Representative distribution strategy": ("magenta", "d"),
    # "Client 4 Disconnect with DIWS": ("green", "s"),
    #"Central substitution": ("brown", "h"),
    "Client 5 Disconnect": ("red", "s"),
    #"Client 1 Disconenct Baseline": ("black", "D"),
    # "Client 2 Disconnect": ("red", "s"),
    #"Client 3 Disconenct Baseline": ("black", "D"),
    # "Client 5 Disconnect with DIWS": ("green", "s"),
    #"Client 1 Substitution": ("green", "p"),
    # "Client 2 Disconnect with DIWS": ("green", "s"),
    #"Client 3 Substitution": ("orange", "v"),
    # "Client 2 & 5 Disconnect": ("red", "s"),
    # "Client 2 & 5 Disconnect with DIWS": ("green", "s"),
    # "IID Client 5 Drop": ("purple", "o"),
    # "Client 2 & 5 IID Node Disconnect": ("red", "o"),
    # "Client 2 & 5 IID Substitution": ("green", "x"),
    # "Multi Epochs": ("black", "s"),
    # "Non-IID MMProx Client 3 Disconnect": ("red", "s"),
    "Client 5 Disconnect with FedAdagrad": ("black", "s")
}

# Extract loss and accuracy data for all experiments
loss_data = {}
accuracy_data = {}

for label, log_file in logs.items():
    loss_data[label] = extract_loss(log_file)
    accuracy_data[label] = extract_accuracy(log_file)

# Plot and save accuracy vs. rounds with a focus on 50% to 70% accuracy
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 6))
for label, (rounds, accuracies) in accuracy_data.items():
    # Filter accuracies and corresponding rounds to focus on 0.5 to 0.7
    # filtered_rounds = [r for r, a in zip(rounds, accuracies) if 0.4 <= a <= 0.7]
    # filtered_accuracies = [a for a in accuracies if 0.4 <= a <= 0.7]
    # plt.plot(filtered_rounds, filtered_accuracies, marker=plot_styles[label][1], color=plot_styles[label][0], label=f"{label} Accuracy")
    plt.plot(rounds, accuracies, marker=plot_styles[label][1], color=plot_styles[label][0], label=f"{label} Accuracy")

plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.ylim(0.4,0.7)
# plt.xlim(7,53)  # Set y-axis limits to 0.5 to 0.7
plt.title("Accuracy vs. Rounds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("niid_mmprox_client_3_drop.png")
plt.close()

print("Focused accuracy plot saved")