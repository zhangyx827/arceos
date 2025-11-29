target remote localhost:1234
b __axplat_main
c
layout split 
set tui mouse-events off
source /home/zyx/.rustup/toolchains/nightly-2025-05-20-x86_64-unknown-linux-gnu/lib/rustlib/etc/gdb_load_rust_pretty_printers.py 
python import sys; sys.path.insert(0, "/home/zyx/.rustup/toolchains/nightly-2025-05-20-x86_64-unknown-linux-gnu/lib/rustlib/etc")