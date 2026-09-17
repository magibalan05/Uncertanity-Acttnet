import sys
import os
import dataset
import model
import metrics

print("System Python Executable:", sys.executable)
print("Checking model compilation...")
m = model.HQUMSNet(in_channels=4, out_channels=1)
print("Model created successfully with parameters:", sum(p.numel() for p in m.parameters()))
