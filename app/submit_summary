#!/bin/bash
# 
# Batch Control Submission script for auto run stats creation.
#
# see:  https://confluence.slac.stanford.edu/display/PSDM/Automatic+Run+Processing
#
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset DISPLAY XAUTHORITY
export PYTHONPATH=
echo "$@"
source /reg/g/psdm/etc/psconda.sh  > /dev/null

SOURCE="${BASH_SOURCE[0]}"
# resolve $SOURCE until the file is no longer a symlink
while [ -h "$SOURCE" ]; do 
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
  # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done

DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

cd $DIR

EXP=$1
shift
RUN=$1
INSTRUMENT=${EXP:0:3}

shift
if [[ $1 ]]; then
  QUEUEREQ=$1
  shift
  if [[ $1 ]]; then
    PYOPT=$1
    shift
    if [[ $1 ]]; then
      RUNFILE=$1
      shift
    fi
  fi
fi

#PYOPT='test'

: ${DIRNAME:="$DIR/../lib/python2.7/site-packages/PyDataSource/"}
if [[ ! -a $DIRNAME ]]; then
    DIRNAME="$DIR/../python/PyDataSource/"
fi

: ${PYFILE:="__main__.py"}
: ${MAX_SIZE:="10001"}
: ${QUEUEREQ:="psanaq"}
#: ${QUEUEREQ="psdebugq"}
OUTDIR="/reg/d/psdm/$INSTRUMENT/$EXP/scratch/nc"
RUNSTR=Run`python "$DIRNAME/get_run_from_runid.py" $EXP $RUN`
EXPRUN="$EXP$RUNSTR"
BATCHUSER=`whoami`
OUTLOG="/reg/neh/home/$BATCHUSER/logs/$EXP/$RUNSTR"
OUTFOLDER="/reg/neh/home/$BATCHUSER/logs/$EXP/"
if [[ ! -a $OUTLOG ]];  then
    mkdir -p $OUTLOG
fi

if [[ $PYOPT == '' ]]; then 
    JOBNAME="$EXPRUN"
    PYOPT='mpi'
    : ${RUNFILE:="$DIRNAME/$PYFILE"}
elif [[ $PYOPT == 'mpi' ]]; then 
    JOBNAME="$EXPRUN"
    PYOPT='mpi'
    : ${RUNFILE:="$DIRNAME/$PYFILE"}
elif [[ $PYOPT == 'build' ]]; then 
    JOBNAME="$EXPRUN"
    : ${RUNFILE:="$DIRNAME/$PYFILE"}
elif [[ $PYOPT == 'beam_stats' ]]; then 
    JOBNAME="$EXPRUN"
    RUNFILE="$DIRNAME/$PYFILE"
elif [[ $PYOPT == 'batch' ]]; then 
    JOBNAME="$EXPRUN"
    : ${RUNFILE:="$DIRNAME/$PYFILE"}
elif [[ $PYOPT == 'epics' ]]; then 
    JOBNAME="$EXP"epics
    : ${RUNFILE:="$DIRNAME/$PYFILE"}
else
    JOBNAME="$EXPRUN"
    EXPFILE="$PYOPT"
    RESDIR="/reg/d/psdm/$INSTRUMENT/$EXP/results"
    if [[ ! -a $RESDIR ]];  then
        RESDIR="/reg/d/psdm/$INSTRUMENT/$EXP/res"
    fi
    RUNFILE="$RESDIR/src/$EXPFILE.py"
    echo 'Run File: '$RUNFILE
    PYOPT='batch'
fi

# Directories not available on psanasvc01 -- need to ssh to 
#: ${CONFIG:=`ssh psana python "$DIRNAME/get_config_file.py" $EXP $RUN`}
#echo 'Config: '$CONFIG 
CONFIG='auto'

echo Processing $EXP Run $RUN
#echo '##' bsub -q "$QUEUEREQ" -J "$JOBNAME" -o $OUTLOG/%J.log python "$RUNFILE" "$PYOPT" --exp="$EXP" --run=$RUN --max_size="$MAX_SIZE" --config="$CONFIG"

LOGDIR=/reg/g/psdm/utils/arp/logs/"$EXP"
if [[ ! -a $LOGDIR ]];  then
    mkdir -p $LOGDIR
fi
LOGSUM="$LOGDIR"/summary_batch_jobs.log

echo '--------------------------------' >> $LOGSUM
date >> $LOGSUM
echo Processing $EXP Run $RUN >> $LOGSUM
echo `uname -a` >> $LOGSUM
echo 'User: '$BATCHUSER >> $LOGSUM
echo 'Run:    '$RUNSTR >> $LOGSUM
echo $EXPRUN >> $LOGSUM
echo 'Log Path: '$OUTLOG >> $LOGSUM
echo 'Config: '$CONFIG >> $LOGSUM
echo 'Run File: '$RUNFILE >> $LOGSUM

if [[ $PYOPT == 'mpi' ]]; then 

    echo 'MPI'
    CHUNKS=11
    if [[ $RUN == *"-"* ]]; then
        #echo '##' bsub -n "$CHUNKS" -J "$EXP_RUN_[$RUN]" -o $OUTLOG/%J.log -q "$QUEUEREQ" mpirun python "$RUNFILE" mpi --exp="$EXP" --run=\$LSB_JOBINDEX --nchunks="$CHUNKS" --config="$CONFIG" --max_size="$MAX_SIZE" 
        #bsub -n "$CHUNKS" -J "$EXP_RUN_[$RUN]" -o $OUTLOG/Run%I/%J.log -q "$QUEUEREQ" mpirun python "$RUNFILE" mpi --exp="$EXP" --run=\$LSB_JOBINDEX --nchunks="$CHUNKS" --config="$CONFIG" --max_size="$MAX_SIZE" 
        echo '##' bsub -n "$CHUNKS" -J [$RUN] -o $OUTFOLDER/Run%I/%J.log -q "$QUEUEREQ" mpirun python "$RUNFILE" mpi --exp="$EXP" --run=\$LSB_JOBINDEX --nchunks="$CHUNKS" --config="$CONFIG" --max_size="$MAX_SIZE" --build="auto"
        bsub -n "$CHUNKS" -J [$RUN] -o $OUTFOLDER/Run%I_%J.log -q "$QUEUEREQ" mpirun python "$RUNFILE" mpi --exp="$EXP" --run=\$LSB_JOBINDEX --nchunks="$CHUNKS" --config="$CONFIG" --max_size="$MAX_SIZE" --build="auto" 
    else
        echo '##' bsub -q "$QUEUEREQ" -n "$CHUNKS" -J "$JOBNAME" -o $OUTLOG/%J.log mpirun python "$RUNFILE" mpi --exp="$EXP" --run="$RUN" --nchunks="$CHUNKS" --config="$CONFIG" --max_size="$MAX_SIZE" 
        bsub -n "$CHUNKS" -J "$JOBNAME" -o $OUTLOG/%J.log -q "$QUEUEREQ" mpirun python "$RUNFILE" mpi --exp="$EXP" --run="$RUN" --nchunks="$CHUNKS" --config="$CONFIG" --max_size="$MAX_SIZE" 
    fi

else 
    echo '##' bsub -q "$QUEUEREQ" -J "$JOBNAME" -o $OUTLOG/%J.log python "$RUNFILE" "$PYOPT" --exp="$EXP" --run=$RUN --max_size="$MAX_SIZE" --config="$CONFIG" >> $LOGSUM

    bsub -q "$QUEUEREQ" -J "$JOBNAME" -o $OUTLOG/%J.log python "$RUNFILE" "$PYOPT" --exp="$EXP" --run=$RUN --max_size="$MAX_SIZE" --config="$CONFIG" --batchuser="$BATCHUSER" 

fi

