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
include deps/eigen/test/CMakeFiles/array_replicate_4.dir/depend.make

# Include the progress variables for this target.
include deps/eigen/test/CMakeFiles/array_replicate_4.dir/progress.make

# Include the compile flags for this target's objects.
include deps/eigen/test/CMakeFiles/array_replicate_4.dir/flags.make

deps/eigen/test/CMakeFiles/array_replicate_4.dir/array_replicate.cpp.o: deps/eigen/test/CMakeFiles/array_replicate_4.dir/flags.make
deps/eigen/test/CMakeFiles/array_replicate_4.dir/array_replicate.cpp.o: ../deps/eigen/test/array_replicate.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object deps/eigen/test/CMakeFiles/array_replicate_4.dir/array_replicate.cpp.o"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/array_replicate_4.dir/array_replicate.cpp.o -c /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/test/array_replicate.cpp

deps/eigen/test/CMakeFiles/array_replicate_4.dir/array_replicate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/array_replicate_4.dir/array_replicate.cpp.i"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/test/array_replicate.cpp > CMakeFiles/array_replicate_4.dir/array_replicate.cpp.i

deps/eigen/test/CMakeFiles/array_replicate_4.dir/array_replicate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/array_replicate_4.dir/array_replicate.cpp.s"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/test/array_replicate.cpp -o CMakeFiles/array_replicate_4.dir/array_replicate.cpp.s

# Object files for target array_replicate_4
array_replicate_4_OBJECTS = \
"CMakeFiles/array_replicate_4.dir/array_replicate.cpp.o"

# External object files for target array_replicate_4
array_replicate_4_EXTERNAL_OBJECTS =

deps/eigen/test/array_replicate_4: deps/eigen/test/CMakeFiles/array_replicate_4.dir/array_replicate.cpp.o
deps/eigen/test/array_replicate_4: deps/eigen/test/CMakeFiles/array_replicate_4.dir/build.make
deps/eigen/test/array_replicate_4: deps/eigen/test/CMakeFiles/array_replicate_4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable array_replicate_4"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/array_replicate_4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
deps/eigen/test/CMakeFiles/array_replicate_4.dir/build: deps/eigen/test/array_replicate_4

.PHONY : deps/eigen/test/CMakeFiles/array_replicate_4.dir/build

deps/eigen/test/CMakeFiles/array_replicate_4.dir/clean:
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/test && $(CMAKE_COMMAND) -P CMakeFiles/array_replicate_4.dir/cmake_clean.cmake
.PHONY : deps/eigen/test/CMakeFiles/array_replicate_4.dir/clean

deps/eigen/test/CMakeFiles/array_replicate_4.dir/depend:
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ggandus/tbnegf/tbnegf/transport/cppbind /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/test /home/ggandus/tbnegf/tbnegf/transport/cppbind/build /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/test /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/test/CMakeFiles/array_replicate_4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : deps/eigen/test/CMakeFiles/array_replicate_4.dir/depend

