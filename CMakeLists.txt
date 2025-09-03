cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerForaLABS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SlicerForaLABS")
set(EXTENSION_CONTRIBUTORS "Harlley Hauradou (Nuclear Engineering Program of UFRJ), Thais Hauradou (Nuclear Engineering Program of UFRJ)")
set(EXTENSION_DESCRIPTION "This is an extension for quantitative pore analysis in foraminifera from 3D segmentations (micro-CT). It computes pore count; minimum/maximum/mean diameter; global metrics (volume and surface porosity, S/V, mean wall thickness). Initially focused on foraminifera but applicable to other porous samples.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SlicerForaLABS.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SlicerForaLABS/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ForaLABS)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
