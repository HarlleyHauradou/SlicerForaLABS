#-----------------------------------------------------------------------------
set(MODULE_NAME ForaLABS)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/HTML/MeasuresReport.html
  Resources/HTML/MeasuresReportExport.html
  Resources/Icons/ForaLABS_icon.png
  Resources/Icons/ForaLABS_lockup.png
  Resources/Icons/ImportTXM.png
  Resources/Icons/Measures.png
  Resources/UI/Measures.ui
  Resources/UI/ImportTXM.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
