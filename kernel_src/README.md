# Segmented Multi-LoRA Multiplication (SMLM) kernel

To meet the needs of Loquetier and support its computational flow design, we redesigned Punica’s kernel and named it the Segmented Multi-LoRA Multiplication (SMLM) kernel. It supports automatic gradient tracking in PyTorch, handles each LoRA linear layer independently for each module, reduces computational overhead, and avoids additional LoRA model preprocessing.

We also fixed some bugs in Punica and applied some bug fixes to Flashinfer from its later commits. Therefore, we didn't add them as submodules. Some modules contribute to the work we are based on, but for the purpose of repository brevity, we removed modules that were not used. These modules and work can be reviewed in submodule repositories.

## Installation

Since we modified the kernel, it is currently only available via build from source.

### Build from source

```bash
# Please install torch before loquetier smlm
pip install ninja numpy torch

# Clone this repo
git clone https://github.com/s3co3wjy5tr2bdfj/Loquetier.git
cd Loquetier
git submodule sync
git submodule update --init

# If you encouter problem while compilation, set TORCH_CUDA_ARCH_LIST to your CUDA architecture.
# export TORCH_CUDA_ARCH_LIST="8.0"

# Build and install loquetier smlm
cd kernel_src
pip install -v --no-build-isolation .
```
