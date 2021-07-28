import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from MADX.madx_tool import Structure
from optimizers.GaussNewton import GaussNewton


now = datetime.now()
structure = Structure()
response_matrix = structure.calculate_response_matrix(structure.structure, structure.structure_in_lines, 1e-4, 0)
print(datetime.now()-now)
plt.plot(structure.twiss_table_4D.s,structure.twiss_table_4D.x,'r')
plt.plot(structure.twiss_table_6D.s,structure.twiss_table_6D.x,'b')
plt.show()

optimizer = GaussNewton()
optimizer.optimize(step=1e-5)
print(datetime.now()-now)
