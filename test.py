from NViT import NViT
import torch
def test_model():
    input_size = (8, 16, 16, 16)
    patch_size = (4, 4, 4, 4)

    num_classes = 10
    dim = 64
    depth = 6
    heads = 8
    mlp_dim = 128

    model = NViT(
        input_size=input_size[1:],  # Exclude batch size
        patch_size=patch_size[1:],
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=input_size[0]
    )

    dummy_input = torch.randn(2, input_size[0], *input_size[1:])  # Batch size 2

    output = model(dummy_input)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
