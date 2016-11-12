import sys
import os

def launchfile(instructions,jobname,launchdir,memory=20,walltime=12,exports='',sourcedir=None):
	stdout = os.path.join(launchdir,jobname,'job.out')
	rundir = os.path.join(launchdir,jobname,'theanomodels')
	if sourcedir is None:
		sourcedir = os.getcwd()
	qsub = """#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime={walltime}:00:00
#PBS -l mem={memory}GB
#PBS -N {jobname}
#PBS -M justinmaojones@nyu.edu
#PBS -j oe
#PBS -o {stdout}

module purge
module load cuda/7.5.18
#module load cudnn/7.5v5.1
module load python/intel/2.7.6 
module load theano/0.8.2
module load ipdb/0.8
module load scipy/intel/0.18.0

SOURCEDIR={sourcedir}
mkdir -p {rundir}
cp -R $SOURCEDIR/* {rundir}
cd {rundir}

export PYTHONPATH=$PYTHONPATH:{rundir}

{exports}

{instructions}
""".format(**locals())

	return qsub

def genjobname(launchdir,prefix='qsub_run'):
	jobnames = sorted([f for f in os.listdir(launchdir) if f[:len(prefix)] == prefix])
	jobnames = [f[len(prefix):len(prefix)+4] for f in jobnames]
	jobnames = [int(f) for f in jobnames if f.isdigit()]
	count = 0
	if len(jobnames) > 0:
		count = max(jobnames)+1
	jobname = '%s%04d' % (prefix,count)
	return jobname


def launch(rootdir,experiment,session,cmd,memory=10,walltime=12):
	#experimentdir = os.path.join(rootdir,experiment)
	launchdir = os.path.join(rootdir,'launch')

	# generate and launch qsub scripts and save to rundir
	#os.system('mkdir -p %s' % experimentdir)
	os.system('mkdir -p %s' % launchdir)

	jobname = genjobname(launchdir) + '_' + session
	rundir = os.path.join(launchdir,jobname)
	os.system('mkdir -p %s' % rundir)
	qsub = launchfile(cmd,jobname,launchdir,memory,walltime,exports='')
	print '\nlaunching... %s' % rundir
	print '$  %s' % cmd 
	runfile = os.path.join(rundir,'run.q')
	with open(runfile,'w') as f:
		f.write(qsub)
	os.system('qsub %s' % runfile)
		





