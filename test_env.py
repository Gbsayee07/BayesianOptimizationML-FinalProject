import numpy as np
import matplotlib.pyplot as plt
import sobol_seq

print("NumPy OK:", np.__version__)
print("Matplotlib OK:", plt)
print("Sobol OK:", sobol_seq.i4_sobol(2, 0))