import os
import sys
import io
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "RealESRGAN_x2_trained_generator.pth"
)

SCALE = 2


# ============================================================
# RESIDUAL DENSE BLOCK
# ============================================================

class ResidualDenseBlock(nn.Module):

    def __init__(
        self,
        num_feat=64,
        num_grow_ch=32
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            num_feat,
            num_grow_ch,
            3,
            1,
            1
        )

        self.conv2 = nn.Conv2d(
            num_feat + num_grow_ch,
            num_grow_ch,
            3,
            1,
            1
        )

        self.conv3 = nn.Conv2d(
            num_feat + 2 * num_grow_ch,
            num_grow_ch,
            3,
            1,
            1
        )

        self.conv4 = nn.Conv2d(
            num_feat + 3 * num_grow_ch,
            num_grow_ch,
            3,
            1,
            1
        )

        self.conv5 = nn.Conv2d(
            num_feat + 4 * num_grow_ch,
            num_feat,
            3,
            1,
            1
        )

        self.lrelu = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(self, x):

        x1 = self.lrelu(
            self.conv1(x)
        )

        x2 = self.lrelu(
            self.conv2(
                torch.cat(
                    (x, x1),
                    dim=1
                )
            )
        )

        x3 = self.lrelu(
            self.conv3(
                torch.cat(
                    (x, x1, x2),
                    dim=1
                )
            )
        )

        x4 = self.lrelu(
            self.conv4(
                torch.cat(
                    (x, x1, x2, x3),
                    dim=1
                )
            )
        )

        x5 = self.conv5(
            torch.cat(
                (x, x1, x2, x3, x4),
                dim=1
            )
        )

        return x5 * 0.2 + x


# ============================================================
# RRDB
# ============================================================

class RRDB(nn.Module):

    def __init__(
        self,
        num_feat=64,
        num_grow_ch=32
    ):
        super().__init__()

        # IMPORTANT:
        # Keep these names exactly as in the checkpoint.
        self.rdb1 = ResidualDenseBlock(
            num_feat=num_feat,
            num_grow_ch=num_grow_ch
        )

        self.rdb2 = ResidualDenseBlock(
            num_feat=num_feat,
            num_grow_ch=num_grow_ch
        )

        self.rdb3 = ResidualDenseBlock(
            num_feat=num_feat,
            num_grow_ch=num_grow_ch
        )

    def forward(self, x):

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return out * 0.2 + x


# ============================================================
# RRDBNET
#
# This is the architecture matching the supplied checkpoint:
#
# Input RGB          : 3 channels
# Pixel-unshuffle x2 : 3 -> 12 channels, H,W -> H/2,W/2
# Features           : 64
# RRDB blocks        : 10
# Growth channels    : 32
# Upsampling stages  : 2
# Output             : 3 channels
#
# Spatial path:
#
# H x W
#   ↓ pixel-unshuffle x2
# H/2 x W/2
#   ↓ upsample x2
# H x W
#   ↓ upsample x2
# 2H x 2W
#
# Therefore:
# 128 x 128 -> 256 x 256
# ============================================================

class RRDBNet(nn.Module):

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=10,
        num_grow_ch=32,
        scale=2
    ):
        super().__init__()

        self.scale = scale

        # The checkpoint has:
        # conv_first.weight = [64, 12, 3, 3]
        self.conv_first = nn.Conv2d(
            num_in_ch * 4,
            num_feat,
            3,
            1,
            1
        )

        self.body = nn.Sequential(
            *[
                RRDB(
                    num_feat=num_feat,
                    num_grow_ch=num_grow_ch
                )
                for _ in range(num_block)
            ]
        )

        self.conv_body = nn.Conv2d(
            num_feat,
            num_feat,
            3,
            1,
            1
        )

        self.conv_up1 = nn.Conv2d(
            num_feat,
            num_feat,
            3,
            1,
            1
        )

        # Required by the supplied checkpoint.
        self.conv_up2 = nn.Conv2d(
            num_feat,
            num_feat,
            3,
            1,
            1
        )

        self.conv_hr = nn.Conv2d(
            num_feat,
            num_feat,
            3,
            1,
            1
        )

        self.conv_last = nn.Conv2d(
            num_feat,
            num_out_ch,
            3,
            1,
            1
        )

        self.lrelu = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Pixel-unshuffle x2
        # ----------------------------------------------------

        feat = F.pixel_unshuffle(
            x,
            downscale_factor=2
        )

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        feat = self.conv_first(feat)

        # ----------------------------------------------------
        # RRDB body
        # ----------------------------------------------------

        body_feat = self.conv_body(
            self.body(feat)
        )

        feat = feat + body_feat

        # ----------------------------------------------------
        # First x2 upsampling
        # ----------------------------------------------------

        feat = F.interpolate(
            feat,
            scale_factor=2,
            mode="nearest"
        )

        feat = self.lrelu(
            self.conv_up1(feat)
        )

        # ----------------------------------------------------
        # Second x2 upsampling
        # ----------------------------------------------------

        feat = F.interpolate(
            feat,
            scale_factor=2,
            mode="nearest"
        )

        feat = self.lrelu(
            self.conv_up2(feat)
        )

        # ----------------------------------------------------
        # HR reconstruction
        # ----------------------------------------------------

        out = self.conv_last(
            self.lrelu(
                self.conv_hr(feat)
            )
        )

        return out


# ============================================================
# NPY -> PNG
#
# This reproduces the user's supplied NPY-to-PNG conversion.
#
# The PNG is created in memory, not left as an extra file.
# Then PIL opens that PNG and converts it to RGB exactly like:
#
# Image.open(...).convert("RGB")
#
# This avoids changing the input data path compared with Colab.
# ============================================================

def npy_to_png_bytes(npy_path):

    # --------------------------------------------------------
    # Load NPY
    # --------------------------------------------------------

    arr = np.load(
        npy_path,
        allow_pickle=False
    )

    # Remove dimensions of size 1
    arr = np.squeeze(arr)

    # --------------------------------------------------------
    # Shape handling
    # --------------------------------------------------------

    if arr.ndim == 2:

        # Grayscale
        pass

    elif arr.ndim == 3:

        # CHW -> HWC
        if (
            arr.shape[0] in [1, 3, 4]
            and arr.shape[2] not in [1, 3, 4]
        ):
            arr = np.transpose(
                arr,
                (1, 2, 0)
            )

        elif arr.shape[2] == 1:

            arr = arr[:, :, 0]

    else:

        raise ValueError(
            f"Unsupported NPY shape: {arr.shape}"
        )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    if arr.dtype != np.uint8:

        arr = arr.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Remove NaN / Inf
        # ----------------------------------------------------

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=255.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Normalize exactly according to supplied code
        # ----------------------------------------------------

        min_val = arr.min()
        max_val = arr.max()

        if min_val >= 0 and max_val <= 1:

            # [0,1] -> [0,255]
            arr = arr * 255.0

        elif min_val >= 0 and max_val <= 255:

            # Already [0,255]
            pass

        else:

            # Arbitrary range -> [0,255]
            if max_val > min_val:

                arr = (
                    (arr - min_val)
                    /
                    (max_val - min_val)
                ) * 255.0

            else:

                arr = np.zeros_like(arr)

        arr = np.clip(
            arr,
            0,
            255
        ).astype(
            np.uint8
        )

    # --------------------------------------------------------
    # Create PIL image exactly as supplied conversion
    # --------------------------------------------------------

    if arr.ndim == 2:

        image = Image.fromarray(
            arr,
            mode="L"
        )

    elif arr.ndim == 3 and arr.shape[2] == 3:

        image = Image.fromarray(
            arr,
            mode="RGB"
        )

    elif arr.ndim == 3 and arr.shape[2] == 4:

        image = Image.fromarray(
            arr,
            mode="RGBA"
        )

    else:

        raise ValueError(
            f"Unsupported array shape: {arr.shape}"
        )

    # --------------------------------------------------------
    # Save PNG to memory
    # --------------------------------------------------------

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer


# ============================================================
# PNG -> MODEL TENSOR
#
# This is the same operation used in the Colab inference:
#
# Image.open(...).convert("RGB")
# np.array(image).astype(np.float32) / 255.0
# HWC -> CHW
# batch dimension
# ============================================================

def png_bytes_to_tensor(
    png_buffer,
    device
):

    # --------------------------------------------------------
    # Open generated PNG
    # --------------------------------------------------------

    image = Image.open(
        png_buffer
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # PNG -> NumPy
    # --------------------------------------------------------

    img = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        img.transpose(
            2,
            0,
            1
        )
    ).unsqueeze(
        0
    ).to(
        device
    )

    return tensor


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model(device):

    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "\nModel checkpoint not found:\n"
            f"{MODEL_PATH}\n\n"
            "Expected repository structure:\n"
            "models/"
            "RealESRGAN_x2_trained_generator.pth"
        )

    print("\nLoading trained generator:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # EXACT MODEL CONFIGURATION
    # --------------------------------------------------------

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=10,
        num_grow_ch=32,
        scale=2
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Select EMA parameters
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "params_ema" in checkpoint:

            state_dict = checkpoint[
                "params_ema"
            ]

            print(
                "Using checkpoint: params_ema"
            )

        elif "params" in checkpoint:

            state_dict = checkpoint[
                "params"
            ]

            print(
                "Using checkpoint: params"
            )

        else:

            state_dict = checkpoint

            print(
                "Using complete state dictionary"
            )

    else:

        state_dict = checkpoint

        print(
            "Using complete state dictionary"
        )

    # --------------------------------------------------------
    # Strict loading
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# PROCESS ONE NPY FILE
# ============================================================

def process_file(
    model,
    input_path,
    output_path,
    device
):

    print(
        f"\nProcessing: "
        f"{os.path.basename(input_path)}"
    )

    # --------------------------------------------------------
    # STEP 1: NPY -> PNG
    # --------------------------------------------------------

    png_buffer = npy_to_png_bytes(
        input_path
    )

    # --------------------------------------------------------
    # STEP 2: PNG -> RGB tensor
    # --------------------------------------------------------

    tensor = png_bytes_to_tensor(
        png_buffer,
        device
    )

    input_h = tensor.shape[2]
    input_w = tensor.shape[3]

    print(
        f"Input PNG size: "
        f"{input_w} x {input_h}"
    )

    # --------------------------------------------------------
    # STEP 3: RRDB inference
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor
        )

    # --------------------------------------------------------
    # STEP 4: CHW -> HWC
    # --------------------------------------------------------

    output = (
        output
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(
            1,
            2,
            0
        )
    )

    # --------------------------------------------------------
    # STEP 5: Clip exactly as Colab
    # --------------------------------------------------------

    output = np.clip(
        output,
        0,
        1
    )

    # --------------------------------------------------------
    # STEP 6: Convert to grayscale
    #
    # The original NPY conversion can create grayscale PNG.
    # Colab then does .convert("RGB"), so the model receives
    # three identical channels.
    #
    # For the NPY submission output, return one grayscale
    # channel.
    # --------------------------------------------------------

    output_gray = np.mean(
        output,
        axis=2
    )

    output_gray = np.clip(
        output_gray,
        0,
        1
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # STEP 7: Check x2 output
    # --------------------------------------------------------

    expected_h = input_h * SCALE
    expected_w = input_w * SCALE

    if output_gray.shape != (
        expected_h,
        expected_w
    ):

        raise RuntimeError(
            "\nUnexpected output size.\n"
            f"Input : {input_h} x {input_w}\n"
            f"Expected: {expected_h} x {expected_w}\n"
            f"Actual  : {output_gray.shape}"
        )

    # --------------------------------------------------------
    # STEP 8: Save restored image as NPY
    #
    # The model output is saved as float32 in [0,1].
    # --------------------------------------------------------

    np.save(
        output_path,
        output_gray
    )

    # --------------------------------------------------------
    # Verify saved file
    # --------------------------------------------------------

    saved = np.load(
        output_path,
        allow_pickle=False
    )

    print(
        f"Output NPY size: "
        f"{saved.shape[1]} x {saved.shape[0]}"
    )

    print(
        f"Output range: "
        f"{saved.min():.6f} "
        f"to "
        f"{saved.max():.6f}"
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Command:
    #
    # python run.py <input-dir> <output-dir>
    # --------------------------------------------------------

    if len(sys.argv) != 3:

        print(
            "\nUsage:"
        )

        print(
            "python run.py <input-dir> <output-dir>"
        )

        sys.exit(1)

    input_dir = os.path.abspath(
        sys.argv[1]
    )

    output_dir = os.path.abspath(
        sys.argv[2]
    )

    # --------------------------------------------------------
    # Check input directory
    # --------------------------------------------------------

    if not os.path.isdir(
        input_dir
    ):

        raise FileNotFoundError(
            f"Input directory not found:\n{input_dir}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "REAL-ESRGAN x2 NPY INFERENCE"
    )

    print(
        "=" * 70
    )

    print(
        "Input folder :",
        input_dir
    )

    print(
        "Output folder:",
        output_dir
    )

    print(
        "Device       :",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU          :",
            torch.cuda.get_device_name(0)
        )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Find all NPY files recursively
    # --------------------------------------------------------

    npy_files = []

    for root, _, files in os.walk(
        input_dir
    ):

        for filename in files:

            # Ignore macOS metadata files
            if filename.startswith(
                "._"
            ):
                continue

            if not filename.lower().endswith(
                ".npy"
            ):
                continue

            input_path = os.path.join(
                root,
                filename
            )

            relative_path = os.path.relpath(
                input_path,
                input_dir
            )

            npy_files.append(
                (
                    input_path,
                    relative_path
                )
            )

    npy_files.sort(
        key=lambda x: x[1]
    )

    if len(npy_files) == 0:

        raise RuntimeError(
            "No .npy files found."
        )

    print(
        f"\nFound {len(npy_files)} .npy files."
    )

    success = 0
    failed = 0

    # --------------------------------------------------------
    # Process every file
    # --------------------------------------------------------

    for index, (
        input_path,
        relative_path
    ) in enumerate(
        npy_files,
        start=1
    ):

        output_path = os.path.join(
            output_dir,
            relative_path
        )

        os.makedirs(
            os.path.dirname(
                output_path
            ),
            exist_ok=True
        )

        print(
            f"\n[{index}/{len(npy_files)}]"
        )

        try:

            process_file(
                model,
                input_path,
                output_path,
                device
            )

            success += 1

        except Exception as error:

            failed += 1

            print(
                "ERROR:",
                relative_path
            )

            print(
                str(error)
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INFERENCE COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "Total   :",
        len(npy_files)
    )

    print(
        "Success :",
        success
    )

    print(
        "Failed  :",
        failed
    )

    print(
        "Output  :",
        output_dir
    )

    print(
        "=" * 70
    )

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

