import collections
import os
import re
from pathlib import Path

import cv2
import numpy
import torch


def extract_specific_module(model, dtype_layer=torch.nn.Linear, name_only=True, verbose=False):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, dtype_layer):
            if verbose:
                print(name, module)
            layers.append((name, module))

    if name_only:
        return [name for name, _ in layers]

    return layers


def get_shape(obj, layer_name):
    if torch.is_tensor(obj):
        return obj.size()
    elif isinstance(obj, (list, tuple)):
        return [get_shape(o, layer_name) for o in obj]
    elif isinstance(obj, collections.OrderedDict):
        return {k: get_shape(v, layer_name) for k, v in obj.items()}
    else:
        raise Exception(
            f"Couldn't deal with layer '{layer_name}': unrecognized data type {type(obj)}"
        )


def custom_torch_summary(model, input_tensor, verbose=2):

    layer_shapes = []
    hooks = []

    for name, layer in model.named_modules():
        # Skip the top-level module
        if layer == model:
            continue

        def hook(module, input, output, layer_name=name):
            # Store the input and output shapes for the module
            try:
                input_shapes = get_shape(input, layer_name)
                output_shapes = get_shape(output, layer_name)

                layer_shapes.append(
                    {
                        "layer_name": layer_name,
                        "input_shapes": input_shapes,
                        "output_shapes": output_shapes,
                        "module": module,
                        "class_name": type(module).__name__,
                    }
                )
            except Exception as e:
                print(f"Error processing module {module}: {e}")
                print(f"Input type: {type(input)}, Output type: {type(output)}")

        # Register the hook with the current layer
        hooks.append(layer.register_forward_hook(hook))

    # Run a forward pass to trigger the hooks
    with torch.no_grad():
        model(input_tensor)

    # Remove all hooks
    for h in hooks:
        h.remove()

    # Optional
    if verbose:
        print("\n\nSummary ===================================================================")

    for i, shapes in enumerate(layer_shapes):
        if i > verbose:
            break

        print(f"Layer: {shapes['layer_name']} - {shapes['class_name']}")
        print(f"  Input shapes: {shapes['input_shapes']}")
        print(f"  Output shapes: {shapes['output_shapes']}\n")

    return layer_shapes


def extract_dimensions(shape, layer_name):

    if isinstance(shape, list):
        elem = shape[0]
        # Get the first element if it's a list, why ? because in the second layer, I observe that they took the first element as input
    elif isinstance(shape, dict):
        elem = list(shape.values())[0]
    elif isinstance(shape, (torch.Size, tuple)) and len(shape) == 4:
        elem = shape
    else:
        raise ValueError(f"Unexpected shape {shape} in layer {layer_name}")

    _, C, H, W = elem

    return C, H, W


def parse_line(line):
    line = line.strip()

    if not line or line.startswith("=") or line.startswith("-"):
        return None

    # Split the line into parts
    parts = re.split(r"\s{2,}", line)
    if len(parts) < 3:
        return None
    layer_info = parts[0]
    output_shape_str = parts[1]
    param_count_str = parts[2]

    # Extract layer type and layer name
    layer_parts = layer_info.split("-")
    if len(layer_parts) < 2:
        return None
    layer_type = layer_parts[0]
    layer_name = "-".join(layer_parts[1:])

    # Extract output shape
    output_shape = eval(output_shape_str.replace("[-1,", "(").replace("]", ")"))
    # Remove commas in parameter count and convert to int
    param_count = int(param_count_str.replace(",", ""))

    return {
        "class_name": layer_type,
        "layer_name": layer_name,
        "output_shapes": output_shape,
        "param_count": param_count,
    }


def reformat_data(data, previous_output_shape):

    layers = []
    for i, line in enumerate(data):

        if line is None:
            continue

        input_shape = previous_output_shape
        output_shape = (1, *line["output_shapes"])

        layers.append(
            {
                "class_name": f"{line['class_name']}",
                "layer_name": f"{line['class_name']}-{line['layer_name']}",
                "input_shapes": input_shape,
                "output_shapes": output_shape,
            }
        )
        # Update the previous output shape
        previous_output_shape = output_shape

    return layers


def filter_conv_layers(data, previous_output_shape):

    conv_layers = []
    total_data = 0

    for i, line in enumerate(data):
        parsed = parse_line(line)

        if parsed is None:
            continue
        if parsed["class_name"] == "Conv2d":
            # The input shape is the previous output shape
            input_shape = previous_output_shape
            output_shape = parsed["output_shapes"]

            # Compute input and output tensor sizes
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
            output_size = output_shape[0] * output_shape[1] * output_shape[2]

            # Sum the sizes
            layer_data = input_size + output_size
            total_data += layer_data

            conv_layers.append(
                {
                    "layer_name": parsed["layer_name"],
                    "class_name": parsed["class_name"],
                    "input_shapes": input_shape,
                    "output_shapes": output_shape,
                    "total_size": layer_data,
                }
            )

            # Update the previous output shape
            previous_output_shape = output_shape
        else:
            # For non-Conv2d layers, just update the previous output shape
            previous_output_shape = parsed["output_shapes"]

    return conv_layers, total_data


def get_gfpgan_path(version="1.4"):

    if version == "1.3":
        model_name = "GFPGANv1.3"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    elif version == "1.4":
        model_name = "GFPGANv1.4"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

    # determine model paths
    model_path = os.path.join("experiments/pretrained_models", model_name + ".pth")
    if not os.path.isfile(model_path):
        model_path = os.path.join("gfpgan/weights", model_name + ".pth")
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    return model_path


def read_img(img_path, verbose=False):

    img_name = os.path.basename(img_path)
    if verbose:
        print(f"Processing {img_name} ...")
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if verbose:
        print(f"{input_img.shape=}")

    return input_img, basename, ext


def save_image(image, path):
    """Save an image to the specified path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)


def save_restored_faces(
    cropped_faces, restored_faces, restored_img, output, basename, suffix, ext, verbose=True
):
    """Save restored faces and comparison images."""
    suffix = f"_{suffix}" if suffix else ""
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        # Save cropped faces from the processed images.
        save_image(cropped_face, output / "cropped_faces" / f"{basename}_{idx:02d}.png")
        # Save restored face
        save_image(restored_face, output / "restored_faces" / f"{basename}_{idx:02d}{suffix}.png")
        # Save comparison image
        cmp_img = numpy.concatenate((cropped_face, restored_face), axis=1)
        save_image(cmp_img, output / "cmp" / f"{basename}_{idx:02d}.png")

    if restored_img is not None:
        extension = ext[1:] if ext == "auto" else ext
        save_image(restored_img, output / "restored_imgs" / f"{basename}{suffix}.{extension}")

    if verbose:
        print(f"Results are in the [{output}] folder.")
