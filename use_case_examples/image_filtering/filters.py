"Filter definitions, with pre-processing, post-processing and compilation methods."

from torch import nn
import json
import numpy as np
import torch
from concrete.ml.torch.numpy_module import NumpyModule
from concrete.ml.onnx.convert import get_equivalent_numpy_forward
from concrete.ml.common.utils import generate_proxy_function
from concrete.numpy.compilation.compiler import Compiler
from concrete.ml.common.debugging.custom_assert import assert_true
from concrete.ml.version import __version__ as CML_VERSION

from common import AVAILABLE_FILTERS


# Add a "black and white" filter
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2277

class _TorchIdentity(nn.Module):
    """Torch identity model."""

    def forward(self, x):
        """Identity forward pass.
        
        Args:
            x (torch.Tensor): The input image.
        
        Returns:
            x (torch.Tensor): The input image.
        """
        return x

class _TorchInverted(nn.Module):
    """Torch inverted model."""

    def forward(self, x):
        """Forward pass for inverting an image's colors.
        
        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The (color) inverted image.
        """
        return 255 - x

class _TorchRotate(nn.Module):
    """Torch rotated model."""

    def forward(self, x):
        """Forward pass for rotating an image.
        
        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The rotated image.
        """
        return x.transpose(0, 1)

class _TorchConv2D(nn.Module):
    """Torch model for applying a single 2D convolution operator on images."""

    def __init__(self, kernel, n_in_channels=3, n_out_channels=3, groups=1):
        """Initializing the filter
        
        Args:
            kernel (np.ndarray): The convolution kernel to consider.
        """
        super().__init__()
        self.kernel = kernel
        self.n_out_channels = n_out_channels
        self.n_in_channels = n_in_channels
        self.groups = groups

    def forward(self, x):
        """Forward pass for filtering the image using a 2D kernel.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The filtered image.
        
        """
        # Define the convolution parameters
        stride = 1
        kernel_shape = self.kernel.shape

        # Ensure the kernel has a proper shape
        # If the kernel has a 1D shape, a (1, 1) kernel is used for each in_channels 
        if len(kernel_shape) == 1:
            kernel = self.kernel.reshape(
                self.n_out_channels, 
                self.n_in_channels // self.groups, 
                1, 
                1,
            )
        
        # Else, if the kernel has a 2D shape, a single (Kw, Kh) kernel is used on all in_channels 
        elif len(kernel_shape) == 2:
            kernel = self.kernel.expand(
                self.n_out_channels, 
                self.n_in_channels // self.groups, 
                kernel_shape[0], 
                kernel_shape[1],
            )
        else:
            raise ValueError(
                "Wrong kernel shape, only 1D or 2D kernels are accepted. Got kernel of shape "
                f"{kernel_shape}"
            )

        return nn.functional.conv2d(x, kernel, stride=stride, groups=self.groups)


class Filter:
    """Filter class used in the app."""

    def __init__(self, image_filter="inverted"):
        """Initializing the filter class using a given filter.

        Most filters can be found at https://en.wikipedia.org/wiki/Kernel_(image_processing).

        Args:
            image_filter (str): The filter to consider. Default to "inverted".
        """
        
        assert_true(
            image_filter in AVAILABLE_FILTERS, 
            f"Unsupported image filter or transformation. Expected one of {*AVAILABLE_FILTERS,}, "
            f"but got {image_filter}"
        )

        self.filter = image_filter
        self.divide = None
        self.repeat_out_channels = False
    
        if image_filter == "identity":
            self.torch_model = _TorchIdentity()

        elif image_filter == "inverted":
            self.torch_model = _TorchInverted()
        
        elif image_filter == "rotate":
            self.torch_model = _TorchRotate()

        elif image_filter == "black and white":
            # Define the grayscale weights (RGB order)
            # These weights were used in PAL and NTSC video systems and can be found at
            # https://en.wikipedia.org/wiki/Grayscale
            # There are initially supposed to be float weights (0.299, 0.587, 0.114), with
            # 0.299 + 0.587 + 0.114 = 1
            # However, since FHE computations require weights to be integers, we first multiply 
            # these by a factor of 1000. The output image's values are then divided by 1000 in 
            # post-processing in order to retrieve the correct result
            kernel = torch.tensor([299, 587, 114])

            self.torch_model = _TorchConv2D(kernel, n_out_channels=1, groups=1)

            # Division value for post-processing
            self.divide = 1000

            # Grayscaled image needs to be put in RGB format for Gradio display
            self.repeat_out_channels = True

        elif image_filter == "blur":
            kernel = torch.ones((3, 3), dtype=torch.int64)

            self.torch_model = _TorchConv2D(kernel, n_out_channels=3, groups=3)

            # Division value for post-processing
            self.divide = 9
        
        elif image_filter == "sharpen":
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ])

            self.torch_model = _TorchConv2D(kernel, n_out_channels=3, groups=3)
        
        elif image_filter == "ridge detection":
            # Make the filter properly grayscaled, as it is commonly used
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2265
        
            kernel = torch.tensor([
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ])

            self.torch_model = _TorchConv2D(kernel, n_out_channels=1, groups=1)

            # Ridge detection is usually displayed as a grayscaled image, which needs to be put in 
            # RGB format for Gradio display
            self.repeat_out_channels = True

        self.onnx_model = None
        self.fhe_circuit = None
    
    def compile(self, inputset, onnx_model=None):
        """Compile the model using an inputset.

        Args:
            inputset (List[np.ndarray]): The set of images to use for compilation
            onnx_model (onnx.ModelProto): The loaded onnx model to consider. If None, it will be 
                generated automatically using a NumpyModule. Default to None.
        """
        # Reshape the inputs found in inputset. This is done because Torch and Numpy don't follow 
        # the same shape conventions.  
        inputset = tuple(
            np.expand_dims(input.transpose(2,0,1), axis=0).astype(np.int64) for input in inputset
        )
        
        # If no onnx model was given, generate a new one.
        if onnx_model is None:
            numpy_module = NumpyModule(
                self.torch_model,
                dummy_input=torch.from_numpy(inputset[0]),
            )
        
            onnx_model = numpy_module.onnx_model

        # Get the proxy function and parameter mappings for initializing the compiler
        self.onnx_model = onnx_model
        numpy_filter = get_equivalent_numpy_forward(onnx_model)

        numpy_filter_proxy, parameters_mapping = generate_proxy_function(
            numpy_filter, ["inputs"]
        )

        compiler = Compiler(
            numpy_filter_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
        )

        # Compile the filter
        self.fhe_circuit = compiler.compile(inputset)

        return self.fhe_circuit

    def pre_processing(self, input_image):
        """Processing that needs to be applied before encryption.

        Args:
            input_image (np.ndarray): The image to pre-process
        
        Returns:
            input_image (np.ndarray): The pre-processed image
        """
        # Reshape the inputs found in inputset. This is done because Torch and Numpy don't follow 
        # the same shape conventions.  
        input_image = np.expand_dims(input_image.transpose(2,0,1), axis=0).astype(np.int64)
        
        return input_image

    def post_processing(self, output_image):
        """Processing that needs to be applied after decryption.

        Args:
            input_image (np.ndarray): The decrypted image to post-process
        
        Returns:
            input_image (np.ndarray): The post-processed image
        """
        # Apply a division if needed
        if self.divide is not None:
            output_image //= self.divide

        # Clip the image's values to proper RGB standards as filters don't handle such constraints 
        output_image = output_image.clip(0, 255)

        # Reshape the inputs found in inputset. This is done because Torch and Numpy don't follow 
        # the same shape conventions.
        output_image = output_image.transpose(0,2,3,1).squeeze(0)
        
        # Grayscaled image needs to be put in RGB format for Gradio display
        if self.repeat_out_channels:
            output_image = output_image.repeat(3, axis=2)

        return output_image

    @classmethod 
    def from_json(cls, json_path):
        """Instantiate a filter using a json file.

        Args:
            json_path (Union[str, pathlib.Path]): Path to the json file.
        
        Returns:
            model (Filter): The instantiated filter class.
        """
        # Load the parameters from the json file
        with open(json_path, "r", encoding="utf-8") as f:
            serialized_processing = json.load(f)

        # Make sure the version in serialized_model is the same as CML_VERSION
        assert_true(
            serialized_processing["cml_version"] == CML_VERSION,
            f"The version of Concrete ML library ({CML_VERSION}) is different "
            f"from the one used to save the model ({serialized_processing['cml_version']}). "
            "Please update to the proper Concrete ML version.",
        )

        # Initialize the model
        model = cls(image_filter=serialized_processing["model_filter"])

        return model

    def to_json(self, path_dir, file_name="serialized_processing"):
        """Export the parameters to a json file.
        
        Args:
            path_dir (Union[str, pathlib.Path]): The path to consider when saving the file.
            file_name (str): The file name
        """
        # Serialize the parameters 
        serialized_processing = {"model_filter": self.filter,}
        serialized_processing = self._clean_dict_types_for_json(serialized_processing)

        # Add the version of the current CML library
        serialized_processing["cml_version"] = CML_VERSION

        # Save the json file
        with open(
            path_dir / f"{file_name}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(serialized_processing, f)
    
    def _clean_dict_types_for_json(self, d: dict) -> dict:
        """Clean all values in the dict to be json serializable.

        Args:
            d (Dict): The dict to clean

        Returns:
            Dict: The cleaned dict
        """
        key_to_delete = []
        for key, value in d.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                d[key] = [self._clean_dict_types_for_json(v) for v in value]
            elif isinstance(value, dict):
                d[key] = self._clean_dict_types_for_json(value)
            elif isinstance(value, (np.generic, np.ndarray)):
                d[key] = d[key].tolist()

        for key in key_to_delete:
            d.pop(key)
        return d
