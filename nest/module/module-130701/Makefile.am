# Automake file for external dynamic modules for NEST
#
# Hans Ekkehard Plesser, April 2008
# Automake file for the Developer Module
# 
# libml_module is built as a normal, installable library.
# It will be installed to $prefix/lib by make install.
# 
# Headers from this directory are not to be installed upon
# make install. They are therefore included in _SOURCES.


# 1. Exchange "my" in "ml_module" with the name of your model below
#    (ten locations).
# 2. Add all .cpp and .h files from your code as *_SOURCES. Header files
#    are given only so that they will be included in the tarball if you
#    run "make dist" on your module.
# 3. The ml_module* stuff creates a module that can be loaded at runtime.
#    It is called ml_module.so.
# 4. The libml_module* stuff creates a library against which NEST can be
#    linked.

libdir= @libdir@/nest

lib_LTLIBRARIES=      ml_module.la libml_module.la

# Need to set this to correct directory!
#NEST_MODELS_DIR := $(shell echo $(HOME))/opt/NEST/dist/nest-2.2.2/models
NEST_MODELS_DIR="/home/mikael/opt/NEST/dist/nest-2.2.2/models"
# Can not have comments between lines when defining ml_module_la_SOURCES
# Also can not have backslash after comment (maybe only if comment is last

ml_module_la_CXXFLAGS= @AM_CXXFLAGS@
ml_module_la_SOURCES =  ml_module.cpp                   ml_module.h         	\
                        bcpnn_connection.cpp            bcpnn_connection.h      \
                      	izhik_cond_exp.cpp           	izhik_cond_exp.h   		\
                        bcpnn_connection_dopamine.cpp   bcpnn_connection_dopamine.h      \
                     	my_aeif_cond_exp.cpp           	my_aeif_cond_exp.h		\
                        iaf_cond_alpha_bias.cpp         iaf_cond_alpha_bias.h	

ml_module_la_LDFLAGS=  -module

libml_module_la_CXXFLAGS= $(ml_module_la_CXXFLAGS) -DLINKED_MODULE
libml_module_la_SOURCES=  $(ml_module_la_SOURCES)

MAKEFLAGS= @MAKE_FLAGS@

AM_CPPFLAGS= @NEST_CPPFLAGS@ \
             @INCLTDL@ \
             -I$(NEST_MODELS_DIR)

.PHONY: install-slidoc

nobase_pkgdata_DATA=\
	sli/ml_module.sli

install-slidoc:
	NESTRCFILENAME=/dev/null $(DESTDIR)$(NEST_PREFIX)/bin/sli --userargs="@HELPDIRS@" $(NEST_PREFIX)/share/nest/sli/install-help.sli

install-data-hook: install-exec install-slidoc

EXTRA_DIST= sli
