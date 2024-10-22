from openvla_dataloader import get_bridge_dataloader
train_dataloader, val_dataloader = get_bridge_dataloader(batch_size=1)
from PIL import Image
print("aaaaa")
for idx,data in enumerate(val_dataloader):
    im = data['pixel_values'][0]
    im.save(f"./{idx}.png")
    if idx == 50:
        break