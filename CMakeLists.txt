cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ForaLABS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/HarlleyHauradou/SlicerForaLABS")
set(EXTENSION_CATEGORY "Shape Analysis")
set(EXTENSION_CONTRIBUTORS "Harlley Hauradou (Nuclear Engineering Program at UFRJ), Thais Hauradou (Nuclear Engineering Program at UFRJ), Simone Pennafirme (Nuclear Engineering Program at UFRJ), Inaya Lima (Nuclear Engineering Program at UFRJ), Catia Barbosa (Graduate Program in Geosciences (Environmental Geochemistry) at UFF)")
set(EXTENSION_DESCRIPTION "This is an extension for quantitative pore analysis in foraminifera from 3D segmentations (micro-CT). It computes pore count; global metrics (volume, total surface area and pore density, S/V, mean wall thickness). Initially focused on foraminifera but applicable to other porous samples.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/HarlleyHauradou/SlicerForaLABS/main/ForaLABS/Resources/Icons/ForaLABS_icon.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/HarlleyHauradou/SlicerForaLABS/main/ForaLABS/Resources/Icons/screenshot.png")
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
