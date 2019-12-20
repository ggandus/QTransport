add_subdirectory(pybind11 EXCLUDE_FROM_ALL)

file(GLOB SOURCES "src/*.cpp")

pybind11_add_module(_cppmodule ${SOURCES})
# pybind11_add_module(_cppmodule src/green.cpp)

target_include_directories(_cppmodule PRIVATE include)
target_link_libraries(_cppmodule PRIVATE cppcore)
set_target_properties(cppcore PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_cppmodule PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${config} config)
        set_target_properties(_cppmodule PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${config} ${CMAKE_SOURCE_DIR})
    endforeach()
endif()
