import yaml
import torch
import torch.nn as nn
from torchsummary import summary
import time
from jax import numpy as jp
import argparse


def load_yaml_config(file_path: str) -> dict:
    """Load the YAML file describing the network configuration."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_network(config: dict) -> nn.Module:
    """
    Given the parsed network configuration, build a PyTorch Sequential model
    and load the weights from the YAML if available.
    """
    layers = []

    # The YAML indicates `in_shape: [null, 6]`, i.e. (batch_size, 6).
    in_features = config["in_shape"][1]

    for layer_cfg in config["layers"]:
        if layer_cfg["type"] == "dense":
            out_features = layer_cfg["shape"][1]

            # Create linear layer
            linear_layer = nn.Linear(in_features, out_features)

            # If 'weights' are provided in the config, load them
            if "weights" in layer_cfg and layer_cfg["weights"] is not None:
                w = torch.tensor(layer_cfg["weights"][0], dtype=torch.float).T
                b = torch.tensor(layer_cfg["weights"][1], dtype=torch.float)

                # Validate shapes
                if list(w.shape) != [out_features, in_features]:
                    raise ValueError(
                        f"Weight shape {list(w.shape)} doesn't match "
                        f"[{out_features}, {in_features}]"
                    )
                if list(b.shape) != [out_features]:
                    raise ValueError(f"Bias shape {list(b.shape)} doesn't match [{out_features}]")

                # Load data into the linear layer
                with torch.no_grad():
                    linear_layer.weight.copy_(w)
                    linear_layer.bias.copy_(b)

            layers.append(linear_layer)

            # Add activation if present
            activation = layer_cfg.get("activation", None)
            if activation is not None:
                activation = activation.lower()
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")

            # Update for next layer
            in_features = out_features

        else:
            raise ValueError(f"Unsupported layer type: {layer_cfg['type']}")

    return nn.Sequential(*layers)


def scale_output(output: torch.Tensor, action_min: float, action_max: float) -> torch.Tensor:
    """
    Given an output in the range of [-1, 1] (assuming final tanh),
    rescale it to [action_min, action_max].

    If your network output is not guaranteed to be in [-1, 1], you can
    skip this or adjust accordingly.
    """
    # Rescale from [-1, 1] → [action_min, action_max]
    return (output + 1) / 2 * (action_max - action_min) + action_min


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run PyTorch inference with a given YAML config.")
    parser.add_argument(
        "--yaml_file_path", type=str, default="params.yaml", help="Path to the YAML config file."
    )
    args = parser.parse_args()

    # Path to your YAML file
    yaml_file_path = args.yaml_file_path

    # 1. Load the YAML config
    config = load_yaml_config(yaml_file_path)

    # 2. Build the PyTorch model
    model = build_network(config)

    # Summarize model
    summary(model, input_size=(1, 6))

    # Run model on test data to ensure consistency with jax model
    test_data = torch.tensor(
        [
            [-0.112, -0.039, 0.852, 5.426, 4.375, 0.0],
            [0.024, 0.608, 1.552, 6.571, 6.256, 0.0],
            [0.106, 1.032, 0.461, 2.92, -1.169, -0.0],
            [0.135, 1.176, 0.209, 0.472, 0.25, 0.0],
            [0.163, 1.097, 0.319, -1.631, 1.739, 0.0],
            [0.189, 0.802, 0.195, -3.803, 1.03, 0.0],
            [0.179, 0.277, -0.275, -6.074, -0.588, -0.0],
            [0.062, -0.553, -1.521, -8.939, -7.375, -0.0],
            [-0.033, -1.243, -0.661, -5.682, -1.598, -0.0],
            [-0.023, -1.626, 0.493, -2.791, 4.622, 0.0],
            [0.024, -1.76, 0.434, -0.37, 0.581, 0.0],
        ]
    )

    # 4. Run a forward pass (inference)
    model.eval()
    with torch.no_grad():
        for i in range(test_data.shape[0]):
            input_data = torch.tensor(test_data[i])
            raw_output = model(input_data)
            print(input_data, raw_output)

    # 5. Scale output to [action_min, action_max] if needed
    #    The final layer in the YAML has `activation: tanh`,
    #    so its output is in [-1, 1].
    action_min = config["action_min"]
    action_max = config["action_max"]
    scaled_output = scale_output(raw_output, action_min, action_max)

    # Print results
    print("Raw output from the network:", raw_output.item())
    print(f"Scaled output to [{action_min}, {action_max}]:", scaled_output.item())

    start_time = time.time()
    for _ in range(1000):
        input_data = torch.randn(1, config["in_shape"][1])
        _ = model(input_data)
    end_time = time.time()
    print(f"Time taken for 1000 iterations: {end_time - start_time} seconds")
