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
CMAKE_SOURCE_DIR = /home/ggandus/transport

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ggandus/transport/build

# Include any dependencies generated for this target.
include cppmodule/CMakeFiles/_cppmodule.dir/depend.make

# Include the progress variables for this target.
include cppmodule/CMakeFiles/_cppmodule.dir/progress.make

# Include the compile flags for this target's objects.
include cppmodule/CMakeFiles/_cppmodule.dir/flags.make

cppmodule/CMakeFiles/_cppmodule.dir/src/green.cpp.o: cppmodule/CMakeFiles/_cppmodule.dir/flags.make
cppmodule/CMakeFiles/_cppmodule.dir/src/green.cpp.o: ../cppmodule/src/green.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ggandus/transport/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cppmodule/CMakeFiles/_cppmodule.dir/src/green.cpp.o"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_cppmodule.dir/src/green.cpp.o -c /home/ggandus/transport/cppmodule/src/green.cpp

cppmodule/CMakeFiles/_cppmodule.dir/src/green.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_cppmodule.dir/src/green.cpp.i"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ggandus/transport/cppmodule/src/green.cpp > CMakeFiles/_cppmodule.dir/src/green.cpp.i

cppmodule/CMakeFiles/_cppmodule.dir/src/green.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_cppmodule.dir/src/green.cpp.s"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ggandus/transport/cppmodule/src/green.cpp -o CMakeFiles/_cppmodule.dir/src/green.cpp.s

cppmodule/CMakeFiles/_cppmodule.dir/src/leadself.cpp.o: cppmodule/CMakeFiles/_cppmodule.dir/flags.make
cppmodule/CMakeFiles/_cppmodule.dir/src/leadself.cpp.o: ../cppmodule/src/leadself.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ggandus/transport/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object cppmodule/CMakeFiles/_cppmodule.dir/src/leadself.cpp.o"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_cppmodule.dir/src/leadself.cpp.o -c /home/ggandus/transport/cppmodule/src/leadself.cpp

cppmodule/CMakeFiles/_cppmodule.dir/src/leadself.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_cppmodule.dir/src/leadself.cpp.i"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ggandus/transport/cppmodule/src/leadself.cpp > CMakeFiles/_cppmodule.dir/src/leadself.cpp.i

cppmodule/CMakeFiles/_cppmodule.dir/src/leadself.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_cppmodule.dir/src/leadself.cpp.s"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ggandus/transport/cppmodule/src/leadself.cpp -o CMakeFiles/_cppmodule.dir/src/leadself.cpp.s

cppmodule/CMakeFiles/_cppmodule.dir/src/main.cpp.o: cppmodule/CMakeFiles/_cppmodule.dir/flags.make
cppmodule/CMakeFiles/_cppmodule.dir/src/main.cpp.o: ../cppmodule/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ggandus/transport/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object cppmodule/CMakeFiles/_cppmodule.dir/src/main.cpp.o"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_cppmodule.dir/src/main.cpp.o -c /home/ggandus/transport/cppmodule/src/main.cpp

cppmodule/CMakeFiles/_cppmodule.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_cppmodule.dir/src/main.cpp.i"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ggandus/transport/cppmodule/src/main.cpp > CMakeFiles/_cppmodule.dir/src/main.cpp.i

cppmodule/CMakeFiles/_cppmodule.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_cppmodule.dir/src/main.cpp.s"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ggandus/transport/cppmodule/src/main.cpp -o CMakeFiles/_cppmodule.dir/src/main.cpp.s

cppmodule/CMakeFiles/_cppmodule.dir/src/self.cpp.o: cppmodule/CMakeFiles/_cppmodule.dir/flags.make
cppmodule/CMakeFiles/_cppmodule.dir/src/self.cpp.o: ../cppmodule/src/self.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ggandus/transport/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object cppmodule/CMakeFiles/_cppmodule.dir/src/self.cpp.o"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_cppmodule.dir/src/self.cpp.o -c /home/ggandus/transport/cppmodule/src/self.cpp

cppmodule/CMakeFiles/_cppmodule.dir/src/self.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_cppmodule.dir/src/self.cpp.i"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ggandus/transport/cppmodule/src/self.cpp > CMakeFiles/_cppmodule.dir/src/self.cpp.i

cppmodule/CMakeFiles/_cppmodule.dir/src/self.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_cppmodule.dir/src/self.cpp.s"
	cd /home/ggandus/transport/build/cppmodule && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ggandus/transport/cppmodule/src/self.cpp -o CMakeFiles/_cppmodule.dir/src/self.cpp.s

# Object files for target _cppmodule
_cppmodule_OBJECTS = \
"CMakeFiles/_cppmodule.dir/src/green.cpp.o" \
"CMakeFiles/_cppmodule.dir/src/leadself.cpp.o" \
"CMakeFiles/_cppmodule.dir/src/main.cpp.o" \
"CMakeFiles/_cppmodule.dir/src/self.cpp.o"

# External object files for target _cppmodule
_cppmodule_EXTERNAL_OBJECTS =

../_cppmodule.cpython-37m-x86_64-linux-gnu.so: cppmodule/CMakeFiles/_cppmodule.dir/src/green.cpp.o
../_cppmodule.cpython-37m-x86_64-linux-gnu.so: cppmodule/CMakeFiles/_cppmodule.dir/src/leadself.cpp.o
../_cppmodule.cpython-37m-x86_64-linux-gnu.so: cppmodule/CMakeFiles/_cppmodule.dir/src/main.cpp.o
../_cppmodule.cpython-37m-x86_64-linux-gnu.so: cppmodule/CMakeFiles/_cppmodule.dir/src/self.cpp.o
../_cppmodule.cpython-37m-x86_64-linux-gnu.so: cppmodule/CMakeFiles/_cppmodule.dir/build.make
../_cppmodule.cpython-37m-x86_64-linux-gnu.so: cppcore/libcppcore.a
../_cppmodule.cpython-37m-x86_64-linux-gnu.so: cppmodule/CMakeFiles/_cppmodule.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ggandus/transport/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared module ../../_cppmodule.cpython-37m-x86_64-linux-gnu.so"
	cd /home/ggandus/transport/build/cppmodule && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_cppmodule.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cppmodule/CMakeFiles/_cppmodule.dir/build: ../_cppmodule.cpython-37m-x86_64-linux-gnu.so

.PHONY : cppmodule/CMakeFiles/_cppmodule.dir/build

cppmodule/CMakeFiles/_cppmodule.dir/clean:
	cd /home/ggandus/transport/build/cppmodule && $(CMAKE_COMMAND) -P CMakeFiles/_cppmodule.dir/cmake_clean.cmake
.PHONY : cppmodule/CMakeFiles/_cppmodule.dir/clean

cppmodule/CMakeFiles/_cppmodule.dir/depend:
	cd /home/ggandus/transport/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ggandus/transport /home/ggandus/transport/cppmodule /home/ggandus/transport/build /home/ggandus/transport/build/cppmodule /home/ggandus/transport/build/cppmodule/CMakeFiles/_cppmodule.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cppmodule/CMakeFiles/_cppmodule.dir/depend

