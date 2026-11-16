"""Compatibility entry point for the sensing postprocessor.

Use ``shared.sensing.postprocessor`` for new Python integrations. The installed
``orchav-sensing`` command and this module command invoke the same workflow.
"""

from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor, main

__all__ = ["PostProcessingConfig", "SensingPostProcessor", "main"]

if __name__ == "__main__":
    main()
