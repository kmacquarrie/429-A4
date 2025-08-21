import argparse
import base64
import os
import sys

from pathlib import Path
from gem5.utils.requires import requires
from gem5.components.boards.x86_board import X86Board
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.cachehierarchies.classic.private_l1_shared_l2_cache_hierarchy import (PrivateL1SharedL2CacheHierarchy,)
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.coherence_protocol import CoherenceProtocol
from gem5.isas import ISA
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import (Resource, BinaryResource)
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent

# This runs a check to ensure the gem5 binary is compiled to X86
requires(
    isa_required=ISA.X86,
    #coherence_protocol_required=CoherenceProtocol.MESI_TWO_LEVEL,
    #kvm_required=True,
)

# Here we setup a Two Level Cache Hierarchy.
cache_hierarchy = PrivateL1SharedL2CacheHierarchy(
    l1d_size="32KiB",
    l1d_assoc=8,
    l1i_size="32KiB",
    l1i_assoc=8,
    l2_size="256KiB",
    l2_assoc=16,
    #num_l2_banks=1,
)

# Setup the system memory.
# Note, by default DDR3_1600 defaults to a size of 8GiB. However, a current
# limitation with the X86 board is it can only accept memory systems up to 3GB.
# As such, we must fix the size.
memory = SingleChannelDDR3_1600("2GiB")

# Here we setup the processor. 
processor = SimpleProcessor(
    # cpu_type=CPUTypes.ATOMIC,
    cpu_type=CPUTypes.O3, # Switch to this line if restoring from existing checkpoint
    num_cores=1,
    isa=ISA.X86,
)

# Here we setup the board. The X86Board allows for Full-System X86 simulations.
board = X86Board(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# This is the command to run after the system has booted. The first `m5 exit`
# will stop the simulation so we can switch the CPU cores from KVM to timing
# and continue the simulation to run the echo command, sleep for a second,
# then, again, call `m5 exit` to terminate the simulation. After simulation
# has ended you may inspect `m5out/system.pc.com_1.device` to see the echo
# output.
command = "sh -c 'm5 checkpoint;m5 readfile > /tmp/gem5.sh && sh /tmp/gem5.sh'"


def addDemoOptions(parser):
    parser.add_argument(
        "-b", "--bench", default=None, help="Benchmark binary to run"
    )
    parser.add_argument(
        "-r", "--restore-cp", default=None, help="Checkpoint to restore from"
    )

parser = argparse.ArgumentParser()
addDemoOptions(parser)

# Parse now so we can override options
args = parser.parse_args()

cp = None
if args.restore_cp is not None :
    if not os.path.isdir(args.restore_cp):
        print("Could not find checkpoint directory", args.restore_cp)
        sys.exit(1)
    cp = Path(os.path.abspath(args.restore_cp))

# Here we set the Full System workload.
# The `set_workload` function for the X86Board takes a kernel, a disk image,
# and, optionally, a the contents of the "readfile". In the case of the
# "x86-ubuntu-18.04-img", a file to be executed as a script after booting the
# system.
board.set_kernel_disk_workload(
    kernel=Resource("x86-linux-kernel-5.4.49",),
    disk_image=Resource("x86-ubuntu-18.04-img"),
    readfile_contents=command,
    checkpoint=cp
)

# Create temp script to run application
if args.bench is not None:
    print("Replacing startup readfile with runscript:", args.bench)
    board.set_binary_to_run(BinaryResource(os.path.abspath(args.bench)),args=[])

simulator = Simulator(
    board=board,
)
simulator.run()
