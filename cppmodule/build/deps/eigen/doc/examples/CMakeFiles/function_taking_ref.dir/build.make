# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/ggandus/anaconda3/bin/cmake

# The command to remove a file.
RM = /home/ggandus/anaconda3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ggandus/tbnegf/tbnegf/transport/cppbind

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ggandus/tbnegf/tbnegf/transport/cppbind/build

# Include any dependencies generated for this target.
include deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/depend.make

# Include the progress variables for this target.
include deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/progress.make

# Include the compile flags for this target's objects.
include deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/flags.make

deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.o: deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/flags.make
deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.o: ../deps/eigen/doc/examples/function_taking_ref.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.o"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.o -c /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/doc/examples/function_taking_ref.cpp

deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.i"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/doc/examples/function_taking_ref.cpp > CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.i

deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.s"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/doc/examples/function_taking_ref.cpp -o CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.s

# Object files for target function_taking_ref
function_taking_ref_OBJECTS = \
"CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.o"

# External object files for target function_taking_ref
function_taking_ref_EXTERNAL_OBJECTS =

deps/eigen/doc/examples/function_taking_ref: deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/function_taking_ref.cpp.o
deps/eigen/doc/examples/function_taking_ref: deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/build.make
deps/eigen/doc/examples/function_taking_ref: deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable function_taking_ref"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/function_taking_ref.dir/link.txt --verbose=$(VERBOSE)
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples && ./function_taking_ref >/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples/function_taking_ref.out

# Rule to build all files generated by this target.
deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/build: deps/eigen/doc/examples/function_taking_ref

.PHONY : deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/build

deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/clean:
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/function_taking_ref.dir/cmake_clean.cmake
.PHONY : deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/clean

deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/depend:
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ggandus/tbnegf/tbnegf/transport/cppbind /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/doc/examples /home/ggandus/tbnegf/tbnegf/transport/cppbind/build /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : deps/eigen/doc/examples/CMakeFiles/function_taking_ref.dir/depend

