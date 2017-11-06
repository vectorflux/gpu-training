Login Instructions

In these examples you will use the GPU cluster Eiger at CSCS. You have all been given accounts on for this purpose. First log in to the front end:

ssh -X username@ela.cscs.ch

From there, log into Eiger itself:

ssh -X username@eiger.cscs.ch

Eiger contains four different types of GPUs: 7 GeForce GTX285, 2 Tesla C1070, 2 Fermi dual M2050 (2.6 GB), and four Fermi dual C2070 (5.4 GB) cards. Please read the system description.

At this point you are only on the Eiger login node, you will need to create an interactive batch job to access the nodes with the GPUs.

qsub -I -l select=1:ncpus=1:mem=1gb:gpu=CARD -l cput=00:30:00,walltime=00:30:00 -q feed@eiger170

This line above requests an interactive session with 1 CPU on a node with a GPU of type CARD, which is one of:

    geforce
    tesla
    fermi 

If you want the M2050 cards, you have to specify the host, e.g.,

qsub -I -l select=1:ncpus=1:mem=1gb:gpu=CARD:host=eiger207 -l cput=00:30:00,walltime=00:30:00 -q feed@eiger170

Once on the node, you must load the CUDA standard toolkit (SDK). The default is SDK 3.2, which is the last stable version.

module load cuda

All the following exercises assume that you have opened an interactive job on a GPU node, and have loaded the CUDA SDK 3.2 
