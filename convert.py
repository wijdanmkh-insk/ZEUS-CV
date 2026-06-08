import torch

model = torch.load("best_zeus_model.pth", map_location="cpu")
model.eval()

example = torch.randn(1, 3, 224, 224)

traced = torch.jit.trace(model, example)
traced._save_for_lite_interpreter("model.ptl")