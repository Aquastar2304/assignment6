"""
Minimal NIfTI-1 reader/writer for Practical Assignment 6.

WHY THIS EXISTS, AND WHY WE DO NOT JUST USE nibabel.load
--------------------------------------------------------
Every volume in this assignment declares `vox_offset = 348`. The NIfTI-1
standard says a single-file (".nii", magic "n+1") image must place its data at
byte 352 or later, so nibabel flags the header and "fixes" it to 352.

That fix is wrong for these files. The arithmetic settles it:

    file size                       13,107,548 bytes
    header                                 348 bytes
    256 x 256 x 50 voxels, float32  13,107,200 bytes
    348 + 13,107,200             =  13,107,548  <- exact match
    352 + 13,107,200             =  13,107,552  <- 4 bytes past end of file

The data genuinely begins at byte 348. Reading from 352 shifts every voxel by
four bytes and silently corrupts the whole volume. So we honour the offset the
header actually records.

A SECOND HEADER TRAP: scl_slope / scl_inter ARE STALE HERE
----------------------------------------------------------
The standard says that when scl_slope is non-zero the stored value should be
rescaled as `raw * scl_slope + scl_inter`. These files carry
slope = 0.00152590 (= 100/65535) and inter = 50.000763, and the lesion masks
carry slope = 0.00392157 (= 1/255).

Applying them destroys the data. Measured on the raw float32 voxels:

    T1 / T2 images   raw range exactly [0, 100]   -> after scaling [50.00, 50.15]
    lesion mask      raw range exactly [0, 1]     -> after scaling [0, 0.0039]

The voxels are already in physical units. The scl_ fields describe an earlier
integer representation that was converted to float32 without clearing them, so
honouring them would collapse all tissue contrast into a 0.15-wide band. We
therefore ignore them by default (`apply_scaling=False`) and record the raw
header values in the report rather than silently applying or silently dropping.
"""
import struct

import numpy as np

# NIfTI datatype code -> numpy dtype string
DTYPES = {
    2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8",
    256: "i1", 512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
}


def read_header(path):
    """Parse the raw 348-byte NIfTI-1 header without any standards 'fixing'."""
    with open(path, "rb") as fh:
        raw = fh.read(348)
    if len(raw) < 348:
        raise ValueError("%s is too short to be a NIfTI-1 file" % path)

    endian = "<" if struct.unpack("<i", raw[0:4])[0] == 348 else ">"
    sizeof_hdr = struct.unpack(endian + "i", raw[0:4])[0]
    if sizeof_hdr != 348:
        raise ValueError("%s: sizeof_hdr is %d, not 348" % (path, sizeof_hdr))

    dim = struct.unpack(endian + "8h", raw[40:56])
    ndim = int(dim[0])
    shape = tuple(int(d) for d in dim[1:ndim + 1])
    datatype = int(struct.unpack(endian + "h", raw[70:72])[0])
    bitpix = int(struct.unpack(endian + "h", raw[72:74])[0])
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    scl_slope, scl_inter = struct.unpack(endian + "2f", raw[112:120])
    magic = raw[344:348]

    if datatype not in DTYPES:
        raise ValueError("%s: unsupported NIfTI datatype code %d" % (path, datatype))

    return {
        "endian": endian, "shape": shape, "ndim": ndim,
        "datatype": datatype, "bitpix": bitpix,
        "zooms": tuple(float(z) for z in pixdim[1:ndim + 1]),
        "vox_offset": vox_offset,
        "scl_slope": float(scl_slope), "scl_inter": float(scl_inter),
        "magic": magic,
    }


def load_nii(path, apply_scaling=False):
    """Return (array, header_dict). The array is C-contiguous float32."""
    h = read_header(path)
    n = int(np.prod(h["shape"]))
    raw = np.fromfile(path, dtype=h["endian"] + DTYPES[h["datatype"]],
                      count=n, offset=h["vox_offset"])
    if raw.size != n:
        raise ValueError("%s: read %d voxels, expected %d" % (path, raw.size, n))

    # NIfTI stores voxels with the first dimension varying fastest.
    arr = raw.reshape(h["shape"], order="F").astype(np.float32)

    slope, inter = h["scl_slope"], h["scl_inter"]
    h["scaling_applied"] = False
    if apply_scaling and slope != 0.0 and not (slope == 1.0 and inter == 0.0):
        arr = arr * np.float32(slope) + np.float32(inter)
        h["scaling_applied"] = True

    return np.ascontiguousarray(arr), h


def save_nii(path, arr, zooms, dtype="f4"):
    """Write a standards-clean single-file NIfTI-1 (data at byte 352)."""
    arr = np.asarray(arr)
    code = {v: k for k, v in DTYPES.items()}[dtype]
    bitpix = int(np.dtype(dtype).itemsize * 8)

    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    dim = [3] + list(arr.shape) + [1] * (7 - len(arr.shape))
    struct.pack_into("<8h", hdr, 40, *dim[:8])
    struct.pack_into("<h", hdr, 70, code)
    struct.pack_into("<h", hdr, 72, bitpix)
    pix = [1.0] + list(zooms) + [1.0] * (7 - len(zooms))
    struct.pack_into("<8f", hdr, 76, *pix[:8])
    struct.pack_into("<f", hdr, 108, 352.0)       # spec-correct for a new file
    struct.pack_into("<f", hdr, 112, 1.0)         # scl_slope
    struct.pack_into("<f", hdr, 116, 0.0)         # scl_inter
    hdr[344:348] = b"n+1\x00"

    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        fh.write(b"\x00" * 4)                      # pad 348 -> 352
        fh.write(np.asfortranarray(arr.astype(dtype)).tobytes(order="F"))
