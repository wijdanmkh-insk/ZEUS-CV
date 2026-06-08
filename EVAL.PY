import torch

MODEL_PATH = "model_int8.pt"

def main():
    print("[INFO] Loading TorchScript model...")

    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()

    print("[INFO] Model loaded")

    # dummy input standar MobileNetV2
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    print("\n========== MODEL INFO ==========")
    print("Output shape:", out.shape)

    if len(out.shape) == 2:
        print("Batch size:", out.shape[0])
        print("Number of classes:", out.shape[1])
    else:
        print("Unexpected output shape:", out.shape)

    print("================================")


if __name__ == "__main__":
    main()