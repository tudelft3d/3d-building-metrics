export DISPLAY=:99.0;
# export VTKI_OFF_SCREEN=True;
export PYVISTA_OFF_SCREEN=true;
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
~/pymesh_new/bin/jupyter-notebook
