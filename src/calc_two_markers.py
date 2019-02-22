
from Scenarios import *
from plot_transformation_covariance import *
from CameraMarkerCalculations import *


sc = Scenarios(0)
cmc = CameraMarkerCalculations(sc)

marker_a = sc.marker_a
marker_b = sc.marker_b
camera_a = sc.camera_a

camera_a_inputs = cmc.generate_solve_pnp_inputs(False, camera_a, marker_a)

camera_a_solved = cmc.solve_pnp(camera_a_inputs)

marker_b_inputs = cmc.generate_solve_pnp_inputs(True, camera_a_solved, marker_b)

marker_b_solved = cmc.solve_pnp(marker_b_inputs)


title = ("Two marker '{}' {}\n" +
         "marker_a:{}\ncamera_a:{}\nmarker_b:{}\nmarker_b_solved:{}").format(
    sc.ident, sc.as_param_str(),
    marker_a.as_rpy_str(),
    camera_a.as_rpy_str(),
    marker_b.as_rpy_str(),
    marker_b_solved.as_rpy_str())

corners_f_image_a = cmc.generate_corners_f_image(camera_a, marker_a)
corners_f_image_b = cmc.generate_corners_f_image(camera_a, marker_b)

plot_view_and_covariance(title, [corners_f_image_a, corners_f_image_b], marker_b_solved)

xxx=6