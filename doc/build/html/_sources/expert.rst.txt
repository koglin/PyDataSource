# to create stats folders run as 

sit_setup  dm-current

dm-create-folders --dirs stats  --mkdir cxij8816

# make idx and smd files

#makeSmallData -e xppf1615 -r 40
#makeSmallData -e xppf1615 -r 41
#makeSmallData -e xppf1615 -r 42

XppMon_Pim1 off -1
display /reg/neh/home/cpo/ana-0.13.12/offbyone/exp\=xppf1615\:run\=42.pdf


#sudo su psdatmgr
#cd /reg/d/psdm/xpp/xppf1615/
#mv ftc/*.smd.xtc xtc/smalldata
#*

#  764  cp /reg/d/psdm/xpp/xpptut15/results/littleData/xppmodules/scripts/makeSmallData ~koglin/conda/PyDataSource/app/
#  765  cp /reg/d/psdm/xpp/xpptut15/results/littleData/xppmodules/scripts/makeSmallData ~koglin/conda/PyDataSource/app/.
#  766  cd
#  767  cd conda/PyDataSource/
#  768  cd app/
#  769  vim makeSmallData 
#  770  ./makeSmallData -e cxig0715 -r 172
#  771  bjobs
#  772  bjobs -wa
#  773  ls /reg/d/psdm/cxi/cxig0715/ftc
#  774  ls /reg/d/psdm/cxi/cxig0715/ftc -l
#  775  ls /reg/d/psdm/cxi/cxig0715/ftc -la
#  776  ls /reg/d/psdm/cxi/cxig0715/ftc/ -la
#  777  ls /reg/d/psdm/cxi/cxig0715/ftc/ -lah
#  778  bjobs
#  779  bjobs -a
#  780  ./makeSmallData -e cxig0715 -r 86
#  781  ls /reg/d/psdm/cxi/cxig0715/xtc/
#  782  ls /reg/d/psdm/cxi/cxig0715/xtc/ -l
#  783  ./makeSmallData -e cxig0715 -r 200
#  784  ./makeSmallData -e cxig0715 -r 201
#  785  bjobs
#  786  ./makeSmallData -e cxig0715 -r 172
#  787  ./makeSmallData -e cxig0715 -r 171
#  788  bjobs
#  789  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/ -l
#  790  ls /reg/d/psdm/cxi/cxig0715/ftc/smalldata/ -l
#  791  ls /reg/d/psdm/cxi/cxig0715/ftc/
#  792  ls /reg/d/psdm/cxi/cxig0715/ftc/ -l
#  793  ./makeSmallData -e cxig0715 -r 59
#  794  bjobs
#  795  bjobs -w
#  796  ./makeSmallData -e cxig0715 -r 202
#  797  bjobs -w
#  798  ls /reg/d/psdm/cxi/cxig0715/ftc/ -l
#  799  bjobs -w
#  800  ls
#  801  ls /reg/d/psdm/cxi/cxig0715/ftc/ -l
#  802  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/ -l
#  803  ls /reg/d/psdm/cxi/cxig0715/ftc/ -l
#  804  mv -r /reg/d/psdm/cxi/cxig0715/ftc/* /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  805  mv /reg/d/psdm/cxi/cxig0715/ftc/* /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  806  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  807  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/*s00-c00*
#  808  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/*s00-c00* -l
#  809  ls /reg/d/psdm/cxi/cxig0715/ftc/ -l
#  810  mv /reg/d/psdm/cxi/cxig0715/ftc/* /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  811  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/e553-r0200-s00-c00.smd.xtc
#  812  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/*s00-c00* -l
#  813  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/*s00-c00* -la
#  814  ls /reg/d/psdm/cxi/cxig0715/ftc/*201* -l
#  815  mv /reg/d/psdm/cxi/cxig0715/ftc/*201* /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  816  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/e553-r0201-s00-c00.smd.xtc
#  817  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/e553-r0201-s00*
#  818  mv /reg/d/psdm/cxi/cxig0715/ftc/*201* /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  819  mv /reg/d/psdm/cxi/cxig0715/ftc/e553-r0201* /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  820  mv /reg/d/psdm/cxi/cxig0715/ftc/e553-r0201-s00-c00.smd.xtc /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  821  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  822  cp /reg/d/psdm/cxi/cxig0715/ftc/e553-r0201-s00-c00.smd.xtc /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  823  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/ -l
#  824  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/md5 -l
#  825  whoami
#  826  cp /reg/d/psdm/cxi/cxig0715/ftc/e553-r0201-s00-c00.smd.xtc /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  827  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/e553-r0201-s00-c00.smd.xtc
#  828  cp /reg/d/psdm/cxi/cxig0715/ftc/e553-r0201-s00-c00.smd.xtc /reg/d/psdm/cxi/cxig0715/xtc/smalldata/.
#  829  rm /reg/d/psdm/cxi/cxig0715/xtc/smalldata/e553-r0201-s00-c00.smd.xtc
#  830  ls /reg/d/psdm/cxi/cxig0715/xtc/*86*
#  831  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/*86*
#  832  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/*0086*
#  833  bjobs -wa
#  834  ls /reg/d/psdm/cxi/cxig0715/xtc/*86*
#  835  ls /reg/d/psdm/cxi/cxig0715/xtc/*0086*
#  836  bjobs -wa | grep r0086
#  837  ls /reg/d/psdm/cxi/cxig0715/ftc/e553-r0086-s02-c00.smd.xtc
#  838  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/*0086*
#  839  mv /reg/d/psdm/cxi/cxig0715/ftc/e553-r0086* reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  840  mv /reg/d/psdm/cxi/cxig0715/ftc/e553-r0086* reg/d/psdm/cxi/cxig0715/xtc/smalldata
#  841  mv /reg/d/psdm/cxi/cxig0715/ftc/e553-r0086* /reg/d/psdm/cxi/cxig0715/xtc/smalldata
#  842  ls /reg/d/psdm/cxi/cxig0715/xtc/smalldata/e553-r0086-s00-c00.smd.xtc
#  843  touch /reg/d/psdm/cxi/cxig0715/xtc/smalldata/e553-r0086-s00-c00.smd.xtc
#  844  ls -la /reg/d/psdm/cxi/cxig0715/xtc/smalldata/e553-r0086-s00-c00.smd.xtc
#  845  ls -la /reg/d/psdm/cxi/cxig0715/xtc/smalldata/
#  846  sudo su psdatmgr
#  847  ls -la /reg/d/psdm/cxi/cxig0715/xtc/smalldata/*s00-c00*
#  848  ls -la /reg/d/psdm/cxi/cxig0715/xtc/*s00-c00*
#  849  ls /reg/d/psdm/cxi/cxig0715/ -la




