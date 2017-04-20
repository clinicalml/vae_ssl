import time
import os
from contextlib import contextmanager
import datetime

def get_new_dirpath_as_timestring(parentdir,suffix=None,leading_zeros_counter=2):
	assert isinstance(leading_zeros_counter,int), 'leading_zeros_counter must be an int'
	assert leading_zeros_counter > 0, 'leading_zeros_counter must be greater than 0'
	time_str = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
	i=0
	pathexists = True
	while pathexists:
		# e.g. if leading_zeros_counter=3, counter_str = '%03d'
		counter_str = '_%' + '0%dd' % leading_zeros_counter
		dirname = time_str 
		if suffix is not None:
			dirname += '_%s' % suffix
		if i > 0:
			dirname += counter_str % i
		pathexists = os.path.exists(os.path.join(parentdir,dirname))
		i+=1
		assert i < 10**leading_zeros_counter, 'counter exceeded 10**leading_zeros_counter-1'
	return os.path.join(parentdir,dirname)

def format_timestr(seconds):
	hours = int(seconds/3600)
	minutes = int((seconds % 3600)/60)
	seconds = int(seconds - hours*3600 - minutes*60)
	return '%dH:%dM:%dS' % (hours,minutes,seconds)
	

@contextmanager
def track_and_print_time(start_str=None,end_str=None):
	start_time = time.time()
	if start_str:
		print start_str
	yield
	duration = datetime.timedelta(seconds=time.time()-start_time)
	print_str = ''
	if end_str:
		print_str += end_str + ' '
	elif start_str:
		print_str += 'finished: ' + start_str + ' '
	print_str += '(duration: %d days, %s)' % (duration.days,format_timestr(duration.seconds))
	print print_str

	


if __name__=='__main__':
	
	#test track_and_print_time 
	with track_and_print_time('start stuff','end stuff'):
		#test get_new_dirpath_as_timestring
		test_dir = '/tmp/test_get_new_dirpath_as_timestring/'
		os.system('mkdir -p %s' % test_dir)	
		os.system('rm -rf %s/*' % test_dir)
		for i in range(10):
			dirpath = get_new_dirpath_as_timestring(test_dir,None,2)
			os.system('mkdir -p %s' % dirpath) 
			print dirpath
		for i in range(10):
			dirpath = get_new_dirpath_as_timestring(test_dir,'suffix',2)
			os.system('mkdir -p %s' % dirpath) 
			print dirpath
		print format_timestr(1234567.2354)
		time.sleep(1)

	with track_and_print_time('just start stuff'):
		time.sleep(1)

	with track_and_print_time():
		print 'not starting any stuff'
		time.sleep(1)

	with track_and_print_time(end_str='only end string'):
		print 'only an end string'
		time.sleep(1)
			

