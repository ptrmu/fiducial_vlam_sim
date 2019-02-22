
from Scenarios import *
from plot_transformation_covariance import *
from CameraMarkerCalculations import *


sc = Scenarios(0)
cmc = CameraMarkerCalculations(sc)

camera_not_marker = False  # False = fixed marker, True = fixed camera
marker_a = sc.marker_a
camera_a = sc.camera_a

camera_a_inputs = cmc.generate_solve_pnp_inputs(camera_not_marker, camera_a, marker_a)

camera_a_solved = cmc.solve_pnp(camera_a_inputs)

title = "Fixed {} '{}', {}\nmarker_a:{}\ncamera_a:{}\ncamera_a_solved:{}".format(
    "camera" if camera_not_marker else "marker",
    sc.ident, sc.as_param_str(),
    marker_a.as_rpy_str(),
    camera_a.as_rpy_str(),
    camera_a_solved.as_rpy_str())

corners_f_image = cmc.generate_corners_f_image(camera_a, marker_a)

plot_view_and_covariance(title, [corners_f_image], camera_a_solved)

xxx=6