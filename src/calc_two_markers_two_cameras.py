
from Scenarios import *
from plot_transformation_covariance import *
from CameraMarkerCalculations import *


sc = Scenarios(0)
cmc = CameraMarkerCalculations(sc)

marker_a = sc.marker_a
marker_b = sc.marker_b
camera_a = sc.camera_a
camera_b = sc.camera_b

camera_aa_inputs = cmc.generate_solve_pnp_inputs(False, camera_a, marker_a)

camera_aa_solved = cmc.solve_pnp(camera_aa_inputs)

marker_ab_inputs = cmc.generate_solve_pnp_inputs(True, camera_aa_solved, marker_b)

marker_ab_solved = cmc.solve_pnp(marker_ab_inputs)

title = ("Two marker camera_a '{}' {}\n" +
         "marker_a:{}\ncamera_a:{}\nmarker_b:{}\nmarker_b_solved:{}").format(
    sc.ident, sc.as_param_str(),
    marker_a.as_rpy_str(),
    camera_a.as_rpy_str(),
    marker_b.as_rpy_str(),
    marker_ab_solved.as_rpy_str())

corners_f_image_aa = cmc.generate_corners_f_image(camera_a, marker_a)
corners_f_image_ab = cmc.generate_corners_f_image(camera_a, marker_b)

plot_view_and_covariance(title, [corners_f_image_aa, corners_f_image_ab], marker_ab_solved, do_show=False)


camera_ba_inputs = cmc.generate_solve_pnp_inputs(False, camera_b, marker_a)

camera_ba_solved = cmc.solve_pnp(camera_ba_inputs)

marker_bb_inputs = cmc.generate_solve_pnp_inputs(True, camera_ba_solved, marker_b)

marker_bb_solved = cmc.solve_pnp(marker_bb_inputs)

title = ("Two marker camera_b '{}' {}\n" +
         "marker_a:{}\ncamera_b:{}\nmarker_b:{}\nmarker_b_solved:{}").format(
    sc.ident, sc.as_param_str(),
    marker_a.as_rpy_str(),
    camera_b.as_rpy_str(),
    marker_b.as_rpy_str(),
    marker_bb_solved.as_rpy_str())

corners_f_image_ba = cmc.generate_corners_f_image(camera_b, marker_a)
corners_f_image_bb = cmc.generate_corners_f_image(camera_b, marker_b)

plot_view_and_covariance(title, [corners_f_image_ba, corners_f_image_bb], marker_bb_solved, do_show=False)


marker_combined_solved = marker_ab_solved.combine(marker_bb_solved)

title = ("Two solutions for marker b '{}' {}\n" +
         "marker_ab:{}\nmarker_bb:{}\nmarker_combined:{}").format(
    sc.ident, sc.as_param_str(),
    marker_ab_solved.as_rpy_str(),
    marker_bb_solved.as_rpy_str(),
    marker_combined_solved.as_rpy_str())

plot_view_and_covariance(title, [corners_f_image_aa, corners_f_image_ab, corners_f_image_ba, corners_f_image_bb],
                         marker_combined_solved)


xxx=6