#!/bin/bash

# Batch jobs might fail if this is not set
#  https://stackoverflow.com/questions/39840184/python-code-crashes-with-cannot-connect-to-x-server-when-detaching-sshtmux-se
unset DISPLAY XAUTHORITY

export PYTHONPATH=

args="$@"

# .  /reg/g/psdm/etc/ana_env.sh

source /reg/g/psdm/bin/conda_setup ""

SOURCE="${BASH_SOURCE[0]}"
# resolve $SOURCE until the file is no longer a symlink
while [ -h "$SOURCE" ]; do 
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
  # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done

DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

# options from xppmodules littledata
# not all are used
while getopts "e:d:r:n:q:j:c:C:M:hxsmb" OPTION
do
    case $OPTION in
    h)
        usage
        exit 1
        ;;
    r) 
        RUN=$OPTARG
        ;;
    e)
        EXP=$OPTARG
        ;;
    n)
        NUMBER_OF_EVENTS=$OPTARG
        ;;
    q)
        QUEUEREQ=$OPTARG
        ;;
    d)
        DIRNAME=$OPTARG
        ;;
    j)
        NJOBS=$OPTARG
        ;;
    s)
        SINGLEJOB=1
        ;;
    b)
        BUILD_HTML=1
        ;;
    m)
        USEMPI=1
        ;;
    x)
        XTCAV=1
        ;;
    f)
        PYFILE=$OPTARG
        ;;    
    c)
        CHUNK=$OPTARG
        ;;
    C)
        CONFIG=$OPTARG
        ;;
    M)
        MAX_SIZE=$OPTARG
        ;;
    ?)
        exit
        ;;
    esac
done

cd $DIR

: ${EXP:=`echo $DIR | awk '{split($0,a,"/"); print a[6]}'`}
INSTRUMENT=${EXP:0:3}
OUTDIR="/reg/d/psdm/$INSTRUMENT/$EXP/scratch/nc"
#PYFILE=RunSummary/src/"$EXP".py
#PYDIR="~/psana/current/"
: ${DIRNAME="$DIR/../python/PyDataSource/"}
: ${PYFILE="__main__.py"}
: ${MAX_SIZE="10001"}
: ${QUEUEREQ="psanaq"}

echo $DIR
if [[ $RUN == *"-"* ]]; then
    echo "Starting Batch"
    : ${CONFIG:=`python "$DIRNAME/$PYFILE" config --exp=$EXP --run=$RUN`}
else
    STEPS=`python "$DIRNAME/$PYFILE" nsteps --exp=$EXP --run=$RUN`
    : ${CONFIG:=`python "$DIRNAME/$PYFILE" config --exp=$EXP --run=$RUN`}
fi

python "$DIRNAME/$PYFILE" configData --exp=$EXP --run=$RUN

#PYTHONPATH=$PYTHONPATH:$PYDIR:$DIR
echo '==============================='
echo 'Starting Batch job'
echo '-------------------------------'
echo "Steps: $STEPS"
echo "outputdir: $OUTDIR"
echo "config: $CONFIG"
echo 'pyfile:' $PYFILE
echo 'pydir:' $DIRNAME
echo 'batchque:' $QUEUEREQ
echo 'Runs:' $RUN
echo '==============================='

if [[ $RUN == *"-"* ]]; then
    CHUNKS=1
    OUTLOG="$OUTDIR/batch_jobs"
else
    : ${CHUNK:="$STEPS"}
    CHUNK_MIN=1
    CHUNK_MAX=$((CHUNK))
    CHUNKS="$((CHUNK_MIN))-$((CHUNK_MAX))"
    OUTLOG="$OUTDIR/mpi_jobs"
fi

if [[ ! -a $OUTLOG ]];  then
    mkdir -p $OUTLOG
fi

#if [ $MAKEBASE ]; then
#
#    bsub -q psanaq -J "$EXP_$RUN_Array[$CHUNKS]" -o "$OUTLOG/"batch_c%I.out python "$DIRNAME/$PYFILE" base --exp="$EXP" --run=$RUN --nchunks=$CHUNK --ichunk=\$LSB_JOBINDEX --config="$CONFIG" --max_size="$MAX_SIZE"
##    bsub -q psanaq -J "$EXP_$RUN_Array[$CHUNKS]" -o "$OUTLOG/"batch_c%I.out python "$DIRNAME/$PYFILE" base --exp="$EXP" --run=$RUN --nchunks=$CHUNK --ichunk=\$LSB_JOBINDEX --config="$CONFIG" --max_size="$MAX_SIZE"
#
#QUEUEREQ=psanaq

if [ $USEMPI ]; then

    echo 'MPI'
    CHUNKS=24
    mpiexec -n "$CHUNKS" python "$DIRNAME/$PYFILE" mpi --exp="$EXP" --run="$RUN" --nchunks="$CHUNKS" --config="$CONFIG" --max_size="$MAX_SIZE" 
    #@mpiexec -n "$CHUNKS" python "$DIRNAME/$PYFILE" mpi --exp="$EXP" --run="$RUN" --nchunks="$CHUNKS" --config="$CONFIG" --max_size="$MAX_SIZE"
    #mpiexec -n 4 python "$DIRNAME/$PYFILE" mpi --exp="$EXP" --run="$RUN" --nchunks=4 --nevents="$NUMBER_OF_EVENTS" --config="$CONFIG" --max_size="$MAX_SIZE"

#elif [ $CHUNK -eq 1 ]; then
elif [[ $RUN == *"-"* ]]; then

    if [ $BUILD_HTML ]; then
        #ONLY Build html 
        PYOPT='build'
    else
        PYOPT='batch'
    fi

    echo bsub -q "$QUEUEREQ" -J "$EXP_RUN_[$RUN]" -o "$OUTLOG/"batch_Run_%I.out python "$DIRNAME/$PYFILE" "$PYOPT" --exp="$EXP" --run=\$LSB_JOBINDEX --max_size="$MAX_SIZE" --config="$CONFIG" 
    bsub -q "$QUEUEREQ" -J "$EXP_RUN_[$RUN]" -o "$OUTLOG/"batch_Run_%I.out python "$DIRNAME/$PYFILE" "$PYOPT" --exp="$EXP" --run=\$LSB_JOBINDEX --max_size="$MAX_SIZE" --config="$CONFIG" 

#    bsub -q psanaq -J "$EXP_RUN_[$RUN]" -o "$OUTLOG/"batch_Run_%I.out python "$DIRNAME/$PYFILE" test --exp="$EXP" --run=\$LSB_JOBINDEX --max_size="$MAX_SIZE" --config="$CONFIG" 

#    echo bsub -q psanaq -J "$EXP_RUN_[$RUN]" -o "$OUTLOG/"batch_Run_%I.out python "$DIRNAME/$PYFILE" xarray --exp="$EXP" --run=\$LSB_JOBINDEX --nevents="$NUMBER_OF_EVENTS" --max_size="$MAX_SIZE"

#    bsub -q psanaq -J "$EXP_RUN_[$RUN]" -o "$OUTLOG/"batch_Run_%I.out python "$DIRNAME/$PYFILE" xarray --exp="$EXP" --run=\$LSB_JOBINDEX  --config="$CONFIG" --nevents="$NUMBER_OF_EVENTS" --max_size="$MAX_SIZE"


else

    echo bsub -q psanaq -J "$EXP_$RUN_Array[$CHUNKS]" -o "$OUTLOG/"batch_c%I.out python "$DIRNAME/$PYFILE" xarray --exp="$EXP" --run=$RUN --nchunks=$CHUNK --ichunk=\$LSB_JOBINDEX --config="$CONFIG" --max_size="$MAX_SIZE"

#    bsub -q psanaq -J "$EXP_$RUN_Array[$CHUNKS]" -o "$OUTLOG/"batch_c%I.out python "$DIRNAME/$PYFILE" xarray --exp="$EXP" --run=$RUN --nchunks=$CHUNK --ichunk=\$LSB_JOBINDEX --config="$CONFIG" --max_size="$MAX_SIZE"

fi

#bsub -q psfehq -J "myArray[$RUN]" -o "$OUTDIR/"myjobs-%I.out python "$PYFILE" --exp="$EXP" --run=\$LSB_JOBINDEX --nevents=100

#python RunSummary/src/"$EXP".py --exp="$EXP" --run="$RUN" --nevents=100
#python RunSummary/src/"$EXP".py --exp="$EXP" --run="$RUN" 

