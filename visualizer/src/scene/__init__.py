"""Load Mitsuba scenes and assemble renderer-neutral geometry payloads.

``io`` owns XML/mesh parsing, ``assembly`` converts parsed meshes into stable
scene entries, and the remaining modules provide material defaults, UV policy,
texture preparation, and renderer-neutral geometry operations.
``visual_surfaces`` applies optional authored face/UV mappings after neutral
loading, keeping visual partitions separate from the source XML and disk cache.
"""
