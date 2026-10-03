"""Load the unchanged torch-scatter 2.1.1 Python reductions without C++ init.
DiffInt uses only scatter_add/mean, which are pure torch.scatter_add_ in 2.1.1.
The C++ extension fails with modern Apple clang (old PyTorch type-traits).
"""
import importlib.util, sys, types, os
from pathlib import Path
source=Path(os.environ['DIFFINT_LEGACY_SCATTER_SOURCE'])
package=types.ModuleType('torch_scatter');package.__path__=[str(source)]
sys.modules['torch_scatter']=package
from torch_scatter.scatter import scatter_add,scatter_mean
package.scatter_add=scatter_add;package.scatter_mean=scatter_mean
