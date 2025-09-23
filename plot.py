import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

log_text = """

INFO :      aggregate_evaluate: received 0 results and 10 failures
INFO :      
INFO :      [SUMMARY]
INFO :      Run finished 10 round(s) in 992.91s
INFO :          History (loss, distributed):
INFO :                  round 1: 2.951601439142499
INFO :                  round 2: 2.260929415559258
INFO :                  round 3: 2.1438362406289517
INFO :                  round 4: 2.036771946844901
INFO :                  round 5: 1.9537265063968063
INFO :                  round 6: 2.7703895395966622
INFO :                  round 7: 3.314770360325658
INFO :                  round 8: 3.314770360325658
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 0.12889879057924888),
INFO :                        (2, 0.16836409929980903),
INFO :                        (3, 0.1817313812858052),
INFO :                        (4, 0.23604922554636112),
INFO :                        (5, 0.2867017774851876),
INFO :                        (6, 0.08817498291182502),
INFO :                        (7, 0.22213967310549776),
INFO :                        (8, 0.22213967310549776)]}
INFO :      
"""

loss_pattern = re.compile(r"round (\d+): ([0-9.]+)")
rounds_loss = [(int(r), float(l)) for r, l in loss_pattern.findall(log_text)]
rounds_loss.sort(key=lambda x: x[0])
rounds = [r for r, _ in rounds_loss]
losses = [l for _, l in rounds_loss]

acc_pattern = re.compile(r"\((\d+),\s*([0-9.]+)\)")
acc_data = [(int(r), float(a)) for r, a in acc_pattern.findall(log_text.split("accuracy")[1])]
acc_data.sort(key=lambda x: x[0])
accuracies = [a for _, a in acc_data]

# ---- ShouldNodeDisconnect logic (copied from client_app.py) ----
def ShouldNodeDisconnect(partition_id, current_round):
    if (partition_id < 2):
        return False
    # For node n, partition_id is n-1
    # start_disconnect = 5, 6, 7 for partition_ids 2, 3, 4
    start_disconnect = (partition_id + 3)
    end_disconnect = 31

    return start_disconnect <= current_round < end_disconnect


fig, ax1 = plt.subplots(figsize=(10,6))

color_loss = 'tab:red'
color_acc = 'tab:blue'

ax1.set_title("Training Progress: Loss & Accuracy vs. Rounds")
ax1.set_xlabel("Round")
ax1.set_ylabel("Loss", color=color_loss)
ax1.plot(rounds, losses, marker='o', color=color_loss, label="Loss")
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.grid(True, which='both', linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  # second y-axis for accuracy
ax2.set_ylabel("Accuracy", color=color_acc)
ax2.plot(rounds, accuracies, marker='s', color=color_acc, label="Accuracy")
ax2.tick_params(axis='y', labelcolor=color_acc)

num_partitions = 5
cmap = plt.get_cmap('tab10')
patches = []

ymin, ymax = ax1.get_ylim()
# Expand vertical span slightly beyond plotted round ticks
x_min = min(rounds) - 0.5
x_max = max(rounds) + 0.5

for pid in range(num_partitions):
    # Find rounds in our plotted range where this partition is considered disconnected
    disconnected_rounds = [r for r in rounds if ShouldNodeDisconnect(pid, r)]
    if not disconnected_rounds:
        continue
    start = min(disconnected_rounds) - 0.5
    end = max(disconnected_rounds) + 0.5
    color = cmap(pid % 10)
    ax1.axvspan(start, end, color=color, alpha=0.12)
    # Mark first disconnect with a dashed vertical line and label
    first_disconnect = min(disconnected_rounds)
    ax1.axvline(first_disconnect, color=color, linestyle='--', alpha=0.7)
    ax1.text(first_disconnect + 0.1, ymax - 0.05*(ymax-ymin), f"P{pid} drop@{first_disconnect}", color=color, fontsize=9, rotation=90, va='top')
    patches.append(mpatches.Patch(color=color, alpha=0.4, label=f"Partition {pid} disconnects from r={min(disconnected_rounds)}"))

lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines = sum([ll[0] for ll in lines_labels], [])
labels = sum([ll[1] for ll in lines_labels], [])
# add partition patches to legend
lines += patches
labels += [p.get_label() for p in patches]

ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0,1.15), ncol=2)

fig.tight_layout()
plt.show()

# saving plot
fig.savefig("training_progress.png")