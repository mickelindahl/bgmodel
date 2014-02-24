#!/bin/sh

echo "Bootstrapping MyModule"

if test -d autom4te.cache ; then
# we must remove this cache, because it
# may screw up things if configure is run for
# different platforms. 
  echo "  -> Removing old automake cache ..."
  rm -rf autom4te.cache
fi

echo "  -> Running aclocal ..."
aclocal

echo "  -> Installing libtool components ..."
if [ `uname -s` = Darwin ] ; then
# libtoolize is glibtoolize on OSX
  LIBTOOLIZE=glibtoolize
else  
  LIBTOOLIZE=libtoolize
fi
$LIBTOOLIZE --force --copy --ltdl

# autoheader must run before automake 
echo "  -> Running autoheader ..."
autoheader

echo "  -> Running automake ..."
automake --foreign --add-missing --force-missing --copy

echo "  -> Running autoconf ..."
autoconf

echo "Done."
