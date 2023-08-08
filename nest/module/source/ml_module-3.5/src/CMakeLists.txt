# CMakeLists.txt
#
# This file is part of the NEST example module.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

# Add all your sources here and specify dependencies, e.g., link libraries
set( MODULE_SOURCES
     ml_module.h ml_module.cpp
     pif_psc_alpha.cpp pif_psc_alpha.h
     drop_odd_spike_connection.h
     step_pattern_builder.h step_pattern_builder.cpp
     recording_backend_socket.h recording_backend_socket.cpp
     izhik_cond_exp.cpp izhik_cond_exp.h
   )

check_have_sfml_audio()
if ( HAVE_SFML_AUDIO )
  list( APPEND MODULE_SOURCES   
        recording_backend_soundclick.h recording_backend_soundclick.cpp
      )
  set( USER_LINK_LIBRARIES sfml-audio )
  add_definitions( -DHAVE_SFML_AUDIO )
else ()
  message( "-- Skip build of recording backend >SoundClick<. sfml-audio is not installed." )
endif ()

if ( BUILD_SHARED_LIBS )
  # When building shared libraries, also create a module for loading at runtime
  # with the `Install` command.
  add_library( ${MODULE_NAME}_module MODULE ${MODULE_SOURCES} )
  target_link_libraries(${MODULE_NAME}_module ${USER_LINK_LIBRARIES})
  set_target_properties( ${MODULE_NAME}_module
      PROPERTIES
      COMPILE_FLAGS "${NEST_CXXFLAGS} -DLTX_MODULE"
      LINK_FLAGS "${NEST_LIBS}"
      PREFIX ""
      OUTPUT_NAME ${MODULE_NAME} )
  install( TARGETS ${MODULE_NAME}_module
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/nest
      )
endif ()

# Build dynamic/static library for standard linking from NEST.
add_library( ${MODULE_NAME}_lib ${MODULE_SOURCES} )
if ( BUILD_SHARED_LIBS )
  # Dynamic libraries are initiated by a `global` variable of the `SLIModule`,
  # which is included, when the flag `LINKED_MODULE` is set.
  target_compile_definitions( ${MODULE_NAME}_lib PRIVATE -DLINKED_MODULE )
endif ()
set_target_properties( ${MODULE_NAME}_lib
    PROPERTIES
    COMPILE_FLAGS "${NEST_CXXFLAGS}"
    LINK_FLAGS "${NEST_LIBS}"
    OUTPUT_NAME ${MODULE_NAME} )

# Install library
install( TARGETS ${MODULE_NAME}_lib DESTINATION ${CMAKE_INSTALL_LIBDIR}/nest )
