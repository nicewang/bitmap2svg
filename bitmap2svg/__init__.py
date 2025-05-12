"""
Bitmap2SVG: A Python library to convert bitmap images to SVG format using Potrace.
"""

try:
    # The C++ extension module is named `_bitmap2svg_cpp` as defined in setup.py and bindings.cpp
    from ._bitmap2svg_cpp import convert_image_to_svg
    _cpp_module_available = True
except ImportError as e:
    _cpp_module_available = False
    _import_error_message = str(e)

    # Fallback function or raise a more informative error
    def convert_image_to_svg(width: int, height: int, pixels, channels: int) -> str:
        """
        Placeholder for convert_image_to_svg.
        The C++ extension module could not be loaded.
        """
        raise ImportError(
            "The C++ extension module '_bitmap2svg_cpp' could not be loaded. "
            "This is necessary for the bitmap-to-SVG conversion. "
            f"Reason: {_import_error_message}. "
            "Please ensure the package was compiled and installed correctly. "
            "You might be missing the Potrace library (libpotrace-dev or similar) "
            "during the build, or the built extension is not found."
        )

__all__ = ["convert_image_to_svg"]

try:
    from ._bitmap2svg_cpp import __version__ as _cpp_version
    __version__ = _cpp_version
except (ImportError, AttributeError):
    __version__ = "0.2.0"


def get_version():
    """Returns the version of the package."""
    return __version__

if not _cpp_module_available:
    print(
        f"WARNING: bitmap2svg C++ extension not loaded (version {__version__}). "
        "Functionality will be unavailable. "
        f"Import error was: {_import_error_message}"
    )
