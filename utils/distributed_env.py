import os
import sys  

class DistributedEnv:
    def __init__(self):
        if 'LOCAL_RANK' in os.environ:
            # Environment variables set by torch.distributed.launch or torchrun
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.global_rank = int(os.environ['RANK'])
        elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
            # Environment variables set by mpirun
            self.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            self.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            self.global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        else:
            sys.exit("Can't find the evironment variables for local rank")

    def print(self, *args, **kwargs):
        print(f"[Rank:{self.global_rank}]", *args, **kwargs)

    def print_master(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)

    def __str__(self):
        return f"global_rank:{self.global_rank}, world_size:{self.world_size}, local_rank:{self.local_rank}"