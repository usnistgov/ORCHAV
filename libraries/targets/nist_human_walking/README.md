# NIST Human Walking Mesh Sequence

This directory contains a curated reconstructed human walking mesh sequence
used by ORCHAV examples that need an articulated passive scatterer.

The sequence comes from the NIST RF Human Walking Dataset workflow: a
28 GHz bistatic walking capture with synchronized camera and LiDAR side
information used to reconstruct an articulated human mesh sequence and
frame-level body keypoints.

The meshes are bundled as example target geometry only, with enough frames for
the included walking-target scenarios. They are not source model files,
motion-capture inputs, or a broad human-motion library.

`target_metadata.json` declares the mesh's visual front direction for
mobility-driven orientation. The sequence faces local `-X`, so ORCHAV applies a
`180 deg` yaw offset when this target uses
`orientation.type: align_motion`. Optional smoothing is configured on that
same orientation model.
