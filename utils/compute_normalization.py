from tqdm import tqdm
from matplotlib import pyplot as plt
import torch

from data import load_data

train_data_list = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']
train_loader, _ = load_data(train_data_list, train_data_list, 16, transform=False)

max_list = []

for i, x in tqdm(enumerate(train_loader), total=len(train_loader)):
    maxs = x.amax(dim=(2, 3))
    max_list.append(maxs)

x = torch.cat(max_list).permute((1, 0))
maxs95 = []
maxs = []
for i, row in enumerate(x):
    sorted_row, _ = torch.sort(row)
    lowest_10 = sorted_row[:10]
    highest_10 = sorted_row[-10:]

    print(f"Row {i}:")
    print("  Lowest 10:", lowest_10)
    print("  Highest 10:", highest_10)
    print("  95th percentile:", sorted_row[int(0.95*8999)])
    print("  50th percentile:", sorted_row[int(0.50*8999)])
    print("  5th percentile:", sorted_row[int(0.05*8999)])
    print()

    maxs.append(sorted_row[-1].item())
    maxs95.append(sorted_row[int(0.95*8999)].item())

print(maxs)
print(maxs95)
    