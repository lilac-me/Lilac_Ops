#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read raw chip identity from npu-smi without importing operator analysis."""

import subprocess


def get_value_from_lines(lines: list[str], key: str) -> str:
    for line in lines:
        line = " ".join(line.split())
        if key in line:
            return line.split(":")[-1].strip()
    return ""


def get_chip_type() -> str:
    try:
        npu_info_lines = subprocess.check_output(["npu-smi", "info", "-l"]).decode().strip().split("\n")
        npu_id = int(get_value_from_lines(npu_info_lines, "NPU ID"))

        board_info_lines = (
            subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(npu_id)])
            .decode()
            .strip()
            .split("\n")
        )
        chip_name = get_value_from_lines(board_info_lines, "Chip Name")

        if not chip_name:
            chip_info_lines = (
                subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(npu_id), "-c", "0"])
                .decode()
                .strip()
                .split("\n")
            )
        else:
            chip_info_lines = board_info_lines

        chip_name = get_value_from_lines(chip_info_lines, "Chip Name")
        chip_type = get_value_from_lines(chip_info_lines, "Chip Type")
        npu_name = get_value_from_lines(chip_info_lines, "NPU Name")

        if "310" in chip_name:
            if not chip_type:
                return ""
            return (chip_type + chip_name).lower()
        if "910" in chip_name:
            if chip_type:
                return (chip_type + chip_name).lower()
            if not npu_name:
                return chip_name.lower()
            return (chip_name + "_" + npu_name).lower()
        if "950" in chip_name:
            if not npu_name:
                return chip_name.lower()
            return (chip_name + "_" + npu_name).lower()
        if "a2g3" in chip_name.lower():
            return chip_name.lower()
        return ""
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return ""
