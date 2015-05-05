#!/bin/sh



HOME_SENDER=/home/jyotika

HOME_HAMBACH=bahuguna@hambach.inm.kfa-juelich.de:/users/bahuguna/2ndManuscript/20paramsfree/checkRelSWAACT/jc1NoEqjc2/



DEST_0=/git/bgmodel/core

DEST_1=/git/bgmodel/scripts_inhibition

DEST_2=/opt/NEST/module/module-130701

DEST_31=/git/bgmodel/nest/module/compile-module-milner.sh

DEST_32=/opt/NEST/module/

DEST_4=/opt/NEST/dist/nest-2.2.2

DEST_51=/results/papers/inhibition/network/supermicro/conn

DEST_52=/results/papers/inhibition/network/milner/conn









#scp -r $HOME_SENDER$DEST_0/ $HOME_MILNER$DEST_0 

#scp -r $HOME_SENDER$DEST_1/ $HOME_MILNER$DEST_1

#scp -r $HOME_SENDER$DEST_2/ $HOME_MILNER$DEST_2

#scp -r $HOME_SENDER$DEST_31 $HOME_MILNER$DEST_32

#scp -r $HOME_SENDER$DEST_4/ $HOME_MILNER$DEST_4



rsync -ravz $HOME_SENDER$DEST_0/ $HOME_MILNER$DEST_0 

rsync -ravz $HOME_SENDER$DEST_1/ $HOME_MILNER$DEST_1

rsync -ravz $HOME_SENDER$DEST_2/ $HOME_MILNER$DEST_2

rsync -ravz $HOME_SENDER$DEST_31 $HOME_MILNER$DEST_32

rsync -ravz $HOME_SENDER$DEST_4/ $HOME_MILNER$DEST_4

rsync -ravz $HOME_SENDER$DEST_51/ $HOME_MILNER$DEST_52
