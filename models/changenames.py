import os

ignore = ['changenames','__init__','randomvariates','special']
with open('changenames.sh','w') as file:
    for f in os.listdir('.'):
        ignored = False
        for i in ignore:
            if i in f:
                ignored = True
        if not ignored:
            key='vae_ssl_'
            if key in f:
                if 'pyc' in f:
                    file.write('mv %s %s\n'%(f,f[len(key):]))
                else:
                    file.write('mv %s %s\n'%(f,f[len(key):]))
