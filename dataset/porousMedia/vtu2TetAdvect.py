import numpy as np
import pyvista as pv

#Cell-wise solution from GCRS
mesh = pv.read(r'solution_5.vtu')

head_line= "NumTetVerts= %d\nx y z" %(mesh.n_points)
np.savetxt("verts.dat", mesh.points,header=head_line, delimiter=" ", fmt="%s",comments="")

head_line= "NumTetCells= %d\nid1 id2 id3 id4" %(mesh.n_cells)
np.savetxt("cells.dat", mesh.cells.reshape(mesh.n_cells,5)[:,1:5], header=head_line, delimiter=" ", fmt="%s",comments="")

head_line= "p u v w"
PUVW=np.hstack([mesh['Pressure'].reshape(mesh.n_cells,1),mesh['Velocity']])
np.savetxt("solutions_cell.dat", PUVW, header=head_line, delimiter=" ", fmt="%s",comments="")