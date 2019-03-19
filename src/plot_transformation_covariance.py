import Transformation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def calc_cov_ellipse(a, b, d):
    s = np.array([[a, b], [b, d]])
    (w, v) = np.linalg.eig(s)
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    return 2 * np.sqrt(w[0]), 2 * np.sqrt(w[1]), angle


class SubPlot:
    def __init__(self, range, offset, rowspan=1, colspan=1):
        self.range = range
        self.offset = offset
        self.rowspan = rowspan
        self.colspan = colspan

    def plot2grid(self, pos):
        plt.subplot2grid(self.range,
                         [self.offset[0] + pos[0], self.offset[1] + pos[1]],
                         rowspan=self.rowspan, colspan=self.colspan)


def _plot_covariance(sub_plot, t_mu_cvec, t_cov, ts_cvec):
    std = np.sqrt(np.diag(t_cov))
    if ts_cvec is not None:
        tnp_dev = ts_cvec - t_mu_cvec

    if ts_cvec is None:
        max_abs = 2.5 * std
    else:
        max_abs = np.max(np.abs(tnp_dev), axis=1)

    for irow in range(6):
        for icol in range(6):
            sub_plot.plot2grid([irow, icol])
            plt.xlim([-max_abs[icol], max_abs[icol]])
            plt.ylim([-max_abs[irow], max_abs[irow]])
            if ts_cvec is not None:
                plt.plot(tnp_dev[icol, :], tnp_dev[irow, :], '.k')

            ax = plt.gca()
            plt.setp(ax.get_xticklabels(), visible=(irow == 5))
            plt.setp(ax.get_yticklabels(), visible=(icol == 0))

            # plot covariance ellipse
            if icol != irow:
                plt.plot(std[icol] * 2, 0., '.r')
                plt.plot(0., std[irow] * 2, '.r')
                width, height, angle = calc_cov_ellipse(t_cov[icol, icol], t_cov[irow, icol], t_cov[irow, irow])
                ellipse = mpl.patches.Ellipse(xy=[0., 0.], width=width * 2, height=height * 2, angle=angle)
                ax = plt.gca()
                ax.add_artist(ellipse)


def plot_transformation_covariance(title, corners_f_image, t_mu_cvec, t_cov, ts_cvec):
    fig = plt.figure()
    fig.suptitle(title)

    # The image representation
    plt.subplot2grid([6, 8], [2, 0], rowspan=2, colspan=2)

    plt.xlim([0, 920])
    plt.ylim([700, 0])

    # show the corners in the image
    for i in range(4):
        plt.plot(corners_f_image[i * 2], corners_f_image[i * 2 + 1], '.r')

    # The covariances.
    tnp = ts_cvec

    means = t_mu_cvec
    tnp_dev = tnp - means

    std = np.sqrt(np.diag(t_cov))
    np_std = np.std(tnp, axis=1)

    maxs = np.max(tnp, axis=1)
    mins = np.min(tnp, axis=1)
    max_abs = np.max(np.abs(tnp_dev), axis=1)

    means_label = ("t_world_xxx mu, std:\n" +
                   "x {0[0]}, {1[0]}\ny {0[1]}, {1[1]}\nz {0[2]}, {1[2]}\n" +
                   "roll {0[3]}, {1[3]}\npitch {0[4]}, {1[4]}\nyaw {0[5]}, {1[5]}").format(means, std)
    plt.subplot2grid([6, 8], [0, 0])
    plt.axis([0, 1, 0, 1])
    plt.text(0, 0.75, means_label, verticalalignment='top')
    ax = plt.gca()
    ax.set_axis_off()

    _plot_covariance(SubPlot([6, 8], [0, 2]), t_mu_cvec, t_cov, ts_cvec)
    # for irow in range(6):
    #     for icol in range(6):
    #         plt.subplot2grid([6, 8], [irow, icol+2])
    #         if True:
    #             plt.xlim([-max_abs[icol], max_abs[icol]])
    #             plt.ylim([-max_abs[irow], max_abs[irow]])
    #             plt.plot(tnp_dev[icol, :], tnp_dev[irow, :], '.k')
    #         else:
    #             plt.xlim([mins[icol], maxs[icol]])
    #             plt.ylim([mins[irow], maxs[irow]])
    #             plt.plot(tnp[icol, :], tnp[irow, :], '.k')
    #
    #         ax = plt.gca()
    #         plt.setp(ax.get_xticklabels(), visible=(irow == 5))
    #         plt.setp(ax.get_yticklabels(), visible=(icol == 0))
    #
    #         plt.plot(std[icol], 0., '.r')
    #         plt.plot(0., std[irow], '.r')
    #
    #         # plot covariance ellipse
    #         if icol != irow:
    #             ax = plt.gca()
    #             width = 2. * std[icol]
    #             height = 2. * std[irow]
    #             width, height, angle = calc_cov_ellipse(t_cov[icol, icol], t_cov[irow, icol], t_cov[irow, irow])
    #             ellipse = mpl.patches.Ellipse(xy=[0., 0.], width=width, height=height, angle=angle)
    #             ax.add_artist(ellipse)

    plt.show()


def _plot_view(sub_plot, corners_f_images):
    sub_plot.plot2grid([0, 0])

    plt.xlim([0, 920])
    plt.ylim([700, 0])

    # show the corners in the image
    for i in range(len(corners_f_images)):
        for r in range(4):
            plt.plot(corners_f_images[i][r * 2], corners_f_images[i][r * 2 + 1], '.r')


def _plot_std_values(sub_plot, cov):
    sub_plot.plot2grid([0, 0])
    std = np.sqrt(np.diag(cov))

    cov_label = ("std:\n" +
                 "x {:5f}\ny {:5f}\nz {:5f}" +
                 "roll {:5f}\npitch {:5f}\nyaw {:5f}").format(std[0], std[1], std[2],
                                                              std[3], std[4], std[5])

    plt.axis([0, 1, 0, 1])
    plt.text(0, 0.75, cov_label, verticalalignment='top')
    ax = plt.gca()
    ax.set_axis_off()


def plot_view_and_covariance(title, corners_f_images, com, do_show=True):
    fig = plt.figure()
    fig.suptitle(title)

    _plot_view(SubPlot([6, 8], [2, 0], rowspan=2, colspan=2), corners_f_images)

    _plot_std_values(SubPlot([6, 8], [0, 0]), com.cov)

    _plot_covariance(SubPlot([6, 8], [0, 2]), com.mu, com.cov, com.samples)

    if do_show:
        plt.show(fig)
