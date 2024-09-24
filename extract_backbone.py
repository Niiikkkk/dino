import torch

model = torch.load('checkpoint.pth', map_location="cpu")
new_state_dict = {}
for name,weights in model["teacher"].items():
    if name.startswith("module.backbone."):
        name = name.replace("module.backbone.", "")
        new_state_dict[name] = weights
torch.save(new_state_dict, 'resnet50.pth')