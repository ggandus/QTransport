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
include deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/depend.make

# Include the progress variables for this target.
include deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/progress.make

# Include the compile flags for this target's objects.
include deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/flags.make

deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.o: deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/flags.make
deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.o: deps/eigen/doc/snippets/compile_Cwise_boolean_xor.cpp
deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.o: ../deps/eigen/doc/snippets/Cwise_boolean_xor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.o"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.o -c /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets/compile_Cwise_boolean_xor.cpp

deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.i"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets/compile_Cwise_boolean_xor.cpp > CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.i

deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.s"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets/compile_Cwise_boolean_xor.cpp -o CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.s

# Object files for target compile_Cwise_boolean_xor
compile_Cwise_boolean_xor_OBJECTS = \
"CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.o"

# External object files for target compile_Cwise_boolean_xor
compile_Cwise_boolean_xor_EXTERNAL_OBJECTS =

deps/eigen/doc/snippets/compile_Cwise_boolean_xor: deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/compile_Cwise_boolean_xor.cpp.o
deps/eigen/doc/snippets/compile_Cwise_boolean_xor: deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/build.make
deps/eigen/doc/snippets/compile_Cwise_boolean_xor: deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable compile_Cwise_boolean_xor"
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_Cwise_boolean_xor.dir/link.txt --verbose=$(VERBOSE)
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets && ./compile_Cwise_boolean_xor >/home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets/Cwise_boolean_xor.out

# Rule to build all files generated by this target.
deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/build: deps/eigen/doc/snippets/compile_Cwise_boolean_xor

.PHONY : deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/build

deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/clean:
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_Cwise_boolean_xor.dir/cmake_clean.cmake
.PHONY : deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/clean

deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/depend:
	cd /home/ggandus/tbnegf/tbnegf/transport/cppbind/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ggandus/tbnegf/tbnegf/transport/cppbind /home/ggandus/tbnegf/tbnegf/transport/cppbind/deps/eigen/doc/snippets /home/ggandus/tbnegf/tbnegf/transport/cppbind/build /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets /home/ggandus/tbnegf/tbnegf/transport/cppbind/build/deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : deps/eigen/doc/snippets/CMakeFiles/compile_Cwise_boolean_xor.dir/depend

