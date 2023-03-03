from pathlib import Path

import torch
from model import CNV


def main():
    checkpoint_path = Path(__file__).parent
    model_path = checkpoint_path / "8_bit_model.pt"
    loaded = torch.load(model_path)
    net = CNV(num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=3, in_ch=3)
    net.load_state_dict(loaded["model_state_dict"])
    torch.save(net.clear_module.state_dict(), checkpoint_path / "clear_module.pt")
    torch.save(net.encrypted_module.state_dict(), checkpoint_path / "encrypted_module.pt")


if __name__ == "__main__":
    main()
