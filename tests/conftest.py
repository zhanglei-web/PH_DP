"""Test-process initialization for the mixed PyTorch/native robotics stack."""

# PyTorch 2.13 imports its optimizer/Dynamo backend lazily. In this environment,
# importing Triton for the first time after Pinocchio/MuJoCo native libraries have
# already been loaded can crash in the dynamic loader. Production SAC constructs
# the Agent before the environment; the full test suite has the opposite import
# order, so preload the same optimizer backend before test-module collection.
import torch._dynamo  # noqa: F401
