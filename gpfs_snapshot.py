"""Emit cumulative client counters for one mounted IBM Storage Scale filesystem."""
import argparse
import json
import subprocess

FIELDS = {"_br_": "client_bytes_read", "_bw_": "client_bytes_written", "_oc_": "open_calls",
          "_cc_": "close_calls", "_rdc_": "read_calls", "_wc_": "write_calls", "_dir_": "readdir_calls",
          "_iu_": "inode_updates"}


def read_snapshot(filesystem, mmpmon):
    output = subprocess.run([mmpmon, "-p"], input="once fs_io_s\n", capture_output=True,
                            text=True, check=True, timeout=8)
    matches = []
    for row in output.stdout.replace("_mmpmon::fs_io_s_", "_fs_io_s_").split("_fs_io_s_")[1:]:
        tokens = row.split()
        fields = dict(zip(tokens[::2], tokens[1::2]))
        if fields.get("_fs_") == filesystem and fields.get("_rc_") == "0":
            matches.append(fields)
    if len(matches) != 1:
        raise ValueError(f"expected one successful fs_io_s row for {filesystem}, got {len(matches)}")
    return {name: int(matches[0][field]) for field, name in FIELDS.items() if field in matches[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filesystem", help="filesystem name from the _fs_ field in mmpmon")
    parser.add_argument("--mmpmon", default="/usr/lpp/mmfs/bin/mmpmon")
    args = parser.parse_args()
    try:
        print(json.dumps(read_snapshot(args.filesystem, args.mmpmon)))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
