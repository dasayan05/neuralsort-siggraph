import os
from condor import condor, Configuration, Job

c = Configuration(request_GPUs=1, request_memory=4096, gpu_memory_range=[4096, 24000], cuda_capability=5.0)
j = Job('python', 'strokesort.py', arguments=dict(
        root='${STORAGE}/datasets/quickdraw',
        embmodel=os.path.join(os.getcwd(), './junks/sketchemb.pth'),
        i=50,
        b=32
    ))

with condor() as session:
    session.submit(j, c)