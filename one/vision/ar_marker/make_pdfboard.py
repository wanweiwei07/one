"""Printable PDF boards: ArUco markers, ChArUco, chessboard.

Every board is drawn on a portrait page (A4 by default) at ``dpi``; sizes are
in mm. Print at 100 % (no "fit to page") and measure the result.
``frame_size`` (width, height) adds a thin rectangle centred on the page as a
cutting guide; None draws nothing.

author: weiwei
date: 20190420
"""

import os

import numpy as np
from PIL import Image
from cv2 import aruco

_MIN_MARGIN_PX = 10             # closest a board may come to the page edge


def _px(mm, dpi):
    """``mm`` in pixels at ``dpi``."""
    return int(round(mm / 25.4 * dpi))


def _page(paper_width, paper_height, dpi, frame_size=None):
    """A white page, with the cutting frame drawn when ``frame_size`` is
    given."""
    page = np.full((_px(paper_height, dpi), _px(paper_width, dpi)), 255,
                   dtype=np.uint8)
    if frame_size is not None:
        w, h = _px(frame_size[0], dpi), _px(frame_size[1], dpi)
        if w + 2 > page.shape[1] or h + 2 > page.shape[0]:
            raise ValueError(f'frame {frame_size} mm does not fit on the '
                             'page')
        lft = (page.shape[1] - w) // 2 - 1
        top = (page.shape[0] - h) // 2 - 1
        rgt, btm = lft + w + 1, top + h + 1
        page[top:btm + 1, [lft, rgt]] = 0
        page[[top, btm], lft:rgt + 1] = 0
    return page


def _paste_centered(page, img, top=0, height=None):
    """Paste ``img`` centred horizontally on ``page`` and vertically in the
    band of rows [top, top + height) (the whole page by default)."""
    if height is None:
        height = page.shape[0] - top
    upper = top + (height - img.shape[0]) // 2
    left = (page.shape[1] - img.shape[1]) // 2
    if upper - top <= _MIN_MARGIN_PX or left <= _MIN_MARGIN_PX:
        raise ValueError('board too large for the page; use fewer or smaller '
                         'squares/markers')
    page[upper:upper + img.shape[0], left:left + img.shape[1]] = img


def _save(page, save_path, name, dpi):
    """Write ``page`` as ``save_path/name.pdf``; returns the path."""
    path = os.path.join(save_path, name + '.pdf')
    Image.fromarray(page).save(path, 'PDF', resolution=dpi)
    return path


def _marker(aruco_dict, marker_id, size_px):
    return aruco.generateImageMarker(aruco_dict, marker_id, size_px)


def _chess_image(nrow, ncolumn, square_px):
    """Chessboard with a black top-left square."""
    cells = (np.add.outer(np.arange(nrow), np.arange(ncolumn)) % 2) * 255
    return np.kron(cells, np.ones((square_px, square_px))).astype(np.uint8)


def _charuco_image(nrow, ncolumn, square_size, marker_ratio, marker_dict,
                   dpi):
    square_px = _px(square_size, dpi)
    board = aruco.CharucoBoard((ncolumn, nrow), square_size,
                               marker_ratio * square_size,
                               aruco.getPredefinedDictionary(marker_dict))
    return board.generateImage((ncolumn * square_px, nrow * square_px))


def make_aruco_board(nrow,
                     ncolumn,
                     marker_dict=aruco.DICT_6X6_250,
                     start_id=0,
                     marker_size=25,
                     save_path='./',
                     name='test',
                     frame_size=None,
                     paper_width=210,
                     paper_height=297,
                     dpi=600):
    """A grid of ``nrow`` x ``ncolumn`` ArUco markers, ids from ``start_id``
    row by row, a quarter marker apart."""
    aruco_dict = aruco.getPredefinedDictionary(marker_dict)
    marker_px = _px(marker_size, dpi)
    pitch = marker_px + marker_px // 4
    board = np.full((nrow * pitch - marker_px // 4,
                     ncolumn * pitch - marker_px // 4), 255, dtype=np.uint8)
    for r in range(nrow):
        for c in range(ncolumn):
            board[r * pitch:r * pitch + marker_px,
                  c * pitch:c * pitch + marker_px] = _marker(
                aruco_dict, start_id + r * ncolumn + c, marker_px)
    page = _page(paper_width, paper_height, dpi, frame_size)
    _paste_centered(page, board)
    return _save(page, save_path, name, dpi)


def make_charuco_board(nrow,
                       ncolumn,
                       marker_dict=aruco.DICT_4X4_250,
                       square_size=25,
                       marker_ratio=.57,
                       save_path='./',
                       name='test',
                       frame_size=None,
                       paper_width=210,
                       paper_height=297,
                       dpi=600):
    """A ChArUco board of ``nrow`` x ``ncolumn`` squares; each marker is
    ``marker_ratio`` of its square. Detect it with
    ``aruco.CharucoBoard((ncolumn, nrow), square_size,
    marker_ratio * square_size, dict)``."""
    page = _page(paper_width, paper_height, dpi, frame_size)
    _paste_centered(page, _charuco_image(nrow, ncolumn, square_size,
                                         marker_ratio, marker_dict, dpi))
    return _save(page, save_path, name, dpi)


def make_dual_marker(marker_dict=aruco.DICT_4X4_250,
                     left_id=0,
                     right_id=1,
                     marker_size=45,
                     save_path='./',
                     name='test',
                     frame_size=(150, 60),
                     paper_width=210,
                     paper_height=297,
                     dpi=600):
    """Two markers side by side, spaced so that the margin to ``frame_size``
    is the same on all four sides."""
    aruco_dict = aruco.getPredefinedDictionary(marker_dict)
    marker_px = _px(marker_size, dpi)
    gap_px = _px(frame_size[0] - frame_size[1] - marker_size, dpi)
    board = np.full((marker_px, 2 * marker_px + gap_px), 255, dtype=np.uint8)
    board[:, :marker_px] = _marker(aruco_dict, left_id, marker_px)
    board[:, -marker_px:] = _marker(aruco_dict, right_id, marker_px)
    page = _page(paper_width, paper_height, dpi, frame_size)
    _paste_centered(page, board)
    return _save(page, save_path, name, dpi)


def make_multi_marker(marker_dict=aruco.DICT_4X4_250,
                      id_list=(0, 1, 2, 3, 4, 5),
                      marker_pos_list=((-80, 30), (-80, 0), (-80, -30),
                                       (80, 30), (80, 0), (80, -30)),
                      marker_size=25,
                      save_path='./',
                      name='test',
                      frame_size=(190, 130),
                      paper_width=210,
                      paper_height=297,
                      dpi=600):
    """Markers ``id_list`` centred at ``marker_pos_list`` (mm from the page
    centre, x to the right, y up)."""
    if len(id_list) != len(marker_pos_list):
        raise ValueError('id_list and marker_pos_list differ in length')
    aruco_dict = aruco.getPredefinedDictionary(marker_dict)
    marker_px = _px(marker_size, dpi)
    page = _page(paper_width, paper_height, dpi, frame_size)
    for marker_id, (x, y) in zip(id_list, marker_pos_list):
        top = page.shape[0] // 2 - _px(y, dpi) - marker_px // 2
        left = page.shape[1] // 2 + _px(x, dpi) - marker_px // 2
        page[top:top + marker_px, left:left + marker_px] = _marker(
            aruco_dict, marker_id, marker_px)
    return _save(page, save_path, name, dpi)


def make_chess_board(nrow,
                     ncolumn,
                     square_size=25,
                     save_path='./',
                     name='test',
                     frame_size=None,
                     paper_width=210,
                     paper_height=297,
                     dpi=600):
    """A chessboard of ``nrow`` x ``ncolumn`` squares. Returns the
    (nrow * ncolumn, 3) grid of square corners in mm, z = 0."""
    page = _page(paper_width, paper_height, dpi, frame_size)
    _paste_centered(page, _chess_image(nrow, ncolumn, _px(square_size, dpi)))
    _save(page, save_path, name, dpi)
    world_points = np.zeros((nrow * ncolumn, 3), np.float32)
    world_points[:, :2] = (np.mgrid[:nrow, :ncolumn].T.reshape(-1, 2)
                           * square_size)
    return world_points


def make_chess_and_charuco_board(nrow_charuco=3,
                                 ncolumn_charuco=5,
                                 square_size_charuco=25,
                                 nrow_chess=3,
                                 ncolumn_chess=5,
                                 square_size_chess=25,
                                 marker_dict=aruco.DICT_6X6_250,
                                 marker_ratio=.57,
                                 save_path='./',
                                 name='test',
                                 frame_size=None,
                                 paper_width=210,
                                 paper_height=297,
                                 dpi=600):
    """A ChArUco board in the upper half of the page, a chessboard in the
    lower half."""
    page = _page(paper_width, paper_height, dpi, frame_size)
    half = page.shape[0] // 2
    _paste_centered(page, _charuco_image(nrow_charuco, ncolumn_charuco,
                                         square_size_charuco, marker_ratio,
                                         marker_dict, dpi),
                    top=0, height=half)
    _paste_centered(page, _chess_image(nrow_chess, ncolumn_chess,
                                       _px(square_size_chess, dpi)),
                    top=half, height=page.shape[0] - half)
    return _save(page, save_path, name, dpi)


if __name__ == '__main__':
    make_multi_marker(marker_dict=aruco.DICT_4X4_250,
                      id_list=[0, 1, 2, 3, 4, 5, 6, 7],
                      marker_pos_list=((-77.5, 37.5), (-77.5, 12.5),
                                       (-77.5, -12.5), (-77.5, -37.5),
                                       (77.5, 37.5), (77.5, 12.5),
                                       (77.5, -12.5), (77.5, -37.5)),
                      marker_size=20,
                      frame_size=(180, 130))
