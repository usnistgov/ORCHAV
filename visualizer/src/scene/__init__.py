"""Load Mitsuba scenes and assemble renderer-neutral geometry payloads.

``io`` owns XML/mesh parsing, ``assembly`` converts parsed meshes into stable
scene entries, and the remaining modules provide material defaults, UV policy,
texture preparation, and renderer-neutral geometry operations.
"""
