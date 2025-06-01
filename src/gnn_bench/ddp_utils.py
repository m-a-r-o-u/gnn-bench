import os
import torch
import torch.distributed as dist


def setup_ddp(args):
    """
    Initialize torch.distributed process group.
    Should be called on each spawned process (one per GPU).
    Expects args.master_addr, args.master_port, args.world_size, args.rank, args.local_rank.
    """
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)

    dist.init_process_group(
        backend=args.distributed_backend,
        rank=args.rank,
        world_size=args.world_size
    )
    torch.cuda.set_device(args.local_rank)


def cleanup_ddp():
    dist.destroy_process_group()
