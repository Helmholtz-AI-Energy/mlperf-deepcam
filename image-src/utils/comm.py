# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import torch
import torch.distributed as dist


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_local_rank():
    """
    Gets node local rank or returns zero if distributed is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    
    #number of GPUs per node
    if torch.cuda.is_available():
        local_rank = dist.get_rank() % torch.cuda.device_count()
    else:
        local_rank = 0
        
    return local_rank


def get_size():
    """
    Gets size of communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size()
    else:
        size = 1
    return size


def get_local_size():
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    if torch.cuda.is_available():
        local_size = torch.cuda.device_count()
    else:
        local_size = 1

    return local_size


def get_local_group(batchnorm_group_size):
    # create local group
    num_groups = get_size() // batchnorm_group_size
    assert (num_groups * batchnorm_group_size == get_size()), "Error, the number of ranks have to be evenly divisible by batchnorm group size"
    my_rank = get_rank()
    world_size = get_size()
    local_group = None
    if world_size > 1 and batchnorm_group_size > 1:
        for i in range(num_groups):
            start = i * batchnorm_group_size
            end = start + batchnorm_group_size
            ranks = list(range(start, end))
            tmp_group = torch.distributed.new_group(ranks = ranks)
            if my_rank in ranks:
                local_group = tmp_group

    return local_group


# split comms using MPI
def init_split(method, instance_size, batchnorm_group_size=1, verbose=False):
    
    # import MPI here:
    from mpi4py import MPI
    
    # get MPI stuff
    mpi_comm = MPI.COMM_WORLD.Dup()
    comm_size = mpi_comm.Get_size()
    comm_rank = mpi_comm.Get_rank()

    # determine the number of instances
    num_instances = comm_size // instance_size
    # determine color dependent on instance id:
    # comm_rank = instance_rank +  instance_id * instance_size
    instance_id = comm_rank // instance_size
    instance_rank = comm_rank % instance_size
    
    # split the communicator
    mpi_instance_comm = mpi_comm.Split(color=instance_id, key=instance_rank)
    
    # for a successful scaffolding, we need to retrieve the IP addresses
    # for each instance_rank == 0 node:
    address = None
    if method == "nccl-slurm": 
        address = os.getenv("SLURM_TOPOLOGY_ADDR").split(".")[-1]
    else:
        raise NotImplementedError(f"Error, wireup method {method} not implemented.")

    #bacst that into to everybody
    address = mpi_instance_comm.bcast(address, root=0)

    # save env vars
    port = "29500"
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = port

    if verbose:
        mpi_comm.barrier()
        print(f"Global Rank: {comm_rank}, Instance Rank: {instance_rank}, Instance ID: {instance_id}, Master Address: {address}", flush=True)
        mpi_comm.barrier()
    
    # do the dist init (if we have non trivial instances)
    if instance_size > 1:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        dist.init_process_group(backend = "nccl",
                                rank = instance_rank,
                                world_size = instance_size)

    # make sure to call a barrier here in order for sharp to use the default comm:
    if dist.is_initialized():
        dist.barrier(device_ids = [get_local_rank()])

    # get the local process group for batchnorm
    batchnorm_group = get_local_group(batchnorm_group_size)

    return mpi_comm, mpi_instance_comm, instance_id, batchnorm_group
    

# do regular init
def init(method, batchnorm_group_size=1):
    #get master address and port
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    if method == "nccl-openmpi":
        addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        #use that URI
        address = addrport.split(":")[0]
        #use the default pytorch port
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK',0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE",0))
        
        #init DDP
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)
        
    elif method == "nccl-slurm":
        rank = int(os.getenv("PMIX_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        #init DDP
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)

    elif method == "nccl-slurm-pmi":
        rank = int(os.getenv("PMI_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port
                                                
        #init DDP
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)
    elif method == "dummy":
        rank = 0
        world_size = 1
        pass
        
    elif method == "mpi":
        #init DDP
        dist.init_process_group(backend = "mpi")
        
    else:
        raise NotImplementedError()
        
    # make sure to call a barrier here in order for sharp to use the default comm:
    if dist.is_initialized():
        dist.barrier(device_ids = [get_local_rank()])


    # get the local process group for batchnorm
    batchnorm_group = get_local_group(batchnorm_group_size)

    return batchnorm_group
