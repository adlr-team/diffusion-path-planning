import numpy as np
import scipy.ndimage as ndimage

from wzk import np2, grid


# Occupancy Image to Distance Image
def img2dist_img(img: np.ndarray, voxel_size: float, add_boundary: bool = True) -> np.ndarray:
    """
    Calculate the signed distance field from an 2D/3D image of the world.
    Obstacles are 1/True, free space is 0/False.
    The distance image is of the same shape as the input image and has positive values outside objects and negative
    values inside objects see 'CHOMP - signed distance field' (10.1177/0278364913488805)
    The voxel_size is used to scale the distance field correctly (the shape of a single pixel / voxel)
    """

    shape = np.array(img.shape)

    if not add_boundary:
        img_bool = img.astype(bool)
        img_int = img.astype(int)

        # Main function
        #                                         # EDT wants objects as 0, rest as 1
        dist_img = ndimage.distance_transform_edt(-img_int + 1, sampling=voxel_size)
        dist_img_complement = ndimage.distance_transform_edt(img_int, sampling=voxel_size)
        dist_img[img_bool] = - dist_img_complement[img_bool]  # Add interior information # noqa

    else:
        # Additional branch, to include boundary filled with obstacles
        img_wb = np.ones(shape + 2, dtype=bool)
        inner_image_idx = np2.slicen(start=np.ones(img.ndim, dtype=int), end=(shape + 1))
        img_wb[inner_image_idx] = img

        dist_img = img2dist_img(img=img_wb, voxel_size=voxel_size, add_boundary=False)
        dist_img = dist_img[inner_image_idx]  # noqa

    return dist_img


def img2dist_img_grad(limits,
                      dist_img=None,  # either
                      img=None, add_boundary=True,  # or
                      ):
    """
    Use Sobel-filter to get the derivative of the edt
    """
    dist_img = helper__dist_img(dist_img=dist_img, img=img, add_boundary=add_boundary, limits=limits)
    voxel_size = grid.limits2voxel_size(shape=dist_img.shape, limits=limits)
    return img2grad(img=dist_img, voxel_size=voxel_size)


def helper__dist_img(dist_img=None,
                     img=None, limits=None, add_boundary=True):
    if dist_img is None:
        voxel_size = grid.limits2voxel_size(shape=img.shape, limits=limits)
        dist_img = img2dist_img(img=img, voxel_size=voxel_size, add_boundary=add_boundary)

    return dist_img


def helper__dist_img_grad(dist_img_grad=None,
                          dist_img=None, add_boundary=True,
                          img=None, limits=None):
    if dist_img_grad is None:
        dist_img_grad = img2dist_img_grad(img=img, add_boundary=add_boundary,
                                          dist_img=dist_img, limits=limits)
    return dist_img_grad


# Image to Distance Function
def img2dist_fun(limits, interp_order=1,
                 img=None, add_boundary=True,  # A
                 dist_img=None,  # B
                 ):
    """
    Interpolate the distance field at each point (continuous).
    The optimizer is happier with the interpolated version, but it is hard to ensure that the interpolation is
    conservative, so the direct variant should be preferred. (actually)
    # DO NOT INTERPOLATE the EDT -> not conservative -> use order=0 / 'nearest'
    """
    dist_img = helper__dist_img(dist_img=dist_img, img=img, limits=limits, add_boundary=add_boundary)

    dist_fun = img2interpolation_fun(img=dist_img, order=interp_order, limits=limits)
    return dist_fun


def img2dist_grad(*,
                  interp_order=1, limits,
                  dist_img_grad=None,  # A
                  img=None, add_boundary=True,  # B 1
                  dist_img=None,  # B 2
                  ):
    dist_img_grad = helper__dist_img_grad(dist_img_grad=dist_img_grad, dist_img=dist_img,
                                          img=img, add_boundary=add_boundary, limits=limits)
    dist_grad = img_grad2interpolation_fun(img_grad=dist_img_grad, order=interp_order, limits=limits)
    return dist_grad


def img2fun_grad(*, dist_img=None, dist_img_grad=None,
                 img=None, add_boundary=True,
                 interp_order_dist, interp_order_grad,
                 limits):
    dist_img = helper__dist_img(dist_img=dist_img, img=img, limits=limits, add_boundary=add_boundary)
    dist_img_grad = helper__dist_img_grad(dist_img_grad=dist_img_grad, dist_img=dist_img,
                                          img=img, add_boundary=add_boundary, limits=limits)

    dist_fun = img2dist_fun(dist_img=dist_img, interp_order=interp_order_dist, limits=limits)
    dist_grad = img2dist_grad(dist_img_grad=dist_img_grad, interp_order=interp_order_grad, limits=limits)

    return dist_fun, dist_grad


# Helper
def __create_radius_temp(radius, shape):
    if np.size(radius) == 1:
        return radius
    d_spheres = np.nonzero(np.array(shape) == np.size(radius))[0][0]
    r_temp_shape = np.ones(len(shape) - 1, dtype=int)
    r_temp_shape[d_spheres] = np.size(radius)
    return radius.reshape(r_temp_shape)


def img2grad(img, voxel_size):
    """
    Calculate the derivative of an image in each direction of the image, using the sobel filter.
    """
    sobel = np.zeros((img.ndim,) + img.shape)
    for d in range(img.ndim):  # Treat image boundary like obstacle
        sobel[d, ...] = ndimage.sobel(img, axis=d, mode="constant", cval=0)

    # Check appropriate scaling of sobel filter, should be correct
    sobel /= 2 ** (2 * img.ndim - 1) * voxel_size  # 2D: 8, 3D: 32
    return sobel


# Image interpolation
def img2interpolation_fun(img, limits, order=1, mode="nearest"):
    """
    Return a function which interpolates between the pixel values of the image (regular spaced grid) by using
    'scipy.ndimage.map_coordinates'. The resulting function takes as input argument either a ndarray or a list of
    world coordinates (!= image coordinates)
    The 'order' keyword indicates which order of interpolation to use. Standard is linear interpolation (order=1).
    For order=0 no interpolation is performed and the value of the nearest grid cell is chosen. Here the values between
    the different cells jump and aren't continuous.

    """

    lower_left = limits[:, 0]
    voxel_size = grid.limits2voxel_size(shape=img.shape, limits=limits)
    factor = 1 / voxel_size

    def interp_fun(x):
        x2 = x.copy()
        if x2.ndim == 1:
            x2 = x2[np.newaxis, :]

        # Map physical coordinates to image indices
        x2 -= lower_left
        x2 *= factor
        x2 -= 0.5

        # print('voxel_size', voxel_size)
        return ndimage.map_coordinates(input=img, coordinates=x2.T, order=order, mode=mode).T

    return interp_fun


def img_grad2interpolation_fun(img_grad, limits, order=1, mode="nearest"):
    """
    Interpolate images representing derivatives (ie from Soble filter). For each dimension there is a derivative /
    layer in the image.
    Return the results combined as an (x, n_dim) array for the derivatives at each point for each dimension.
    """

    n_dim = img_grad.shape[0]

    fun_list = []
    for d in range(n_dim):
        fun_list.append(img2interpolation_fun(img=img_grad[d, ...], order=order, mode=mode, limits=limits))

    def fun_grad(x):
        res = np.empty_like(x)
        for _d in range(n_dim):
            res[..., _d] = fun_list[_d](x=x)

        return res

    return fun_grad


def tries():
    limits = np.array([[-1, 2],
                       [-1, 2]])

    img = np.zeros((20, 20), dtype=bool)
    voxel_size = 3 / 20
    img[3:5, 5:8] = 1
    img[range(20), range(20)] = 1

    dimg = img2dist_img(img=img, voxel_size=voxel_size, add_boundary=False)
    fun = img2interpolation_fun(img=dimg, limits=limits, order=0)


    x = np.array(np.meshgrid(*[np.linspace(limits[i, 0], limits[i, 1], 201) for i in range(2)], indexing="ij"))
    x = x.reshape(2, -1).T
    d = fun(x)

    from wzk import mpl2
    fig, ax = mpl2.new_fig(aspect=1)
    eps = 0.01
    mpl2.imshow(ax=ax, limits=limits, img=img, mask=~img, cmap="k")
    ax.plot(*x[d < eps].T, marker="o", color="red", ls="", markersize=1)
    ax.plot(*x[d >= eps].T, marker="o", color="blue", ls="", markersize=1)

    #
    dimg_grad = img2grad(img=dimg, voxel_size=voxel_size)
    fig, ax = mpl2.new_fig(aspect=1, n_cols=4)

    mpl2.imshow(ax=ax[0], limits=limits, img=img, cmap="Greys")
    mpl2.imshow(ax=ax[1], limits=limits, img=dimg, cmap="Greys")
    mpl2.imshow(ax=ax[2], limits=limits, img=dimg_grad[0, :, :], cmap="Greys")
    mpl2.imshow(ax=ax[3], limits=limits, img=dimg_grad[1, :, :], cmap="Greys")


if __name__ == "__main__":
    tries()
