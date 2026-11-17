import os
import asyncio
import datetime

SLEEP_TIME = 1
RANK_SIZE = 8
LOG_FILE = "npu_util.log"

if os.path.exists(LOG_FILE):
    i = 0
    new_name = LOG_FILE.split(".")[0] + "_" + str(i) + "." + LOG_FILE.split(".")[1]
    while os.path.exists(new_name):
        i += 1
        new_name = LOG_FILE.split(".")[0] + "_" + str(i) + "." + LOG_FILE.split(".")[1]
    os.rename(LOG_FILE, new_name)

async def async_get_one_npu_util(rank):
    proc = await asyncio.create_subprocess_shell(
        f"npu-smi info -t usages -i {rank} | grep Aicore | awk -F: \'{{ print $2}}\'",
        stdout=asyncio.subprocess.PIPE
        )
    stdout, stderr = await proc.communicate()
    result = stdout.decode("UTF-8").strip()
    return result


async def async_print_one_npu_util(rank):
    proc = await asyncio.create_subprocess_shell(
        f"npu-smi info -t usages -i {rank} | grep Aicore | awk -F: \'{{ print $2}}\'",
        stdout=asyncio.subprocess.PIPE
        )
    stdout, stderr = await proc.communicate()
    result = stdout.decode("UTF-8").strip()
    print("{time}, rank={rank}, util={util}".format(time=datetime.datetime.now(), rank=rank, util=result))


async def async_save_one_npu_util(rank, file):
    proc = await asyncio.create_subprocess_shell(
        f"npu-smi info -t usages -i {rank} | grep Aicore | awk -F: \'{{ print $2}}\'",
        stdout=asyncio.subprocess.PIPE
        )
    stdout, stderr = await proc.communicate()
    result = stdout.decode("UTF-8").strip()
    with open(file, "a") as f:
        f.write(f"{datetime.datetime.now()}, rank={rank}, util={result}\n")


async def async_get_all_npu_util():
    tasks = [async_get_one_npu_util(i) for i in range(RANK_SIZE)]
    results = await asyncio.gather(*tasks)
    return results
    

async def async_print_all_npu_util():
    tasks = [async_print_one_npu_util(i) for i in range(RANK_SIZE)]
    await asyncio.gather(*tasks)


async def async_save_all_npu_util():
    tasks = [async_save_one_npu_util(i, LOG_FILE) for i in range(RANK_SIZE)]
    results = await asyncio.gather(*tasks)
    return results


async def main():
    while True:
        tasks = []
        tasks.append(async_print_all_npu_util())
        tasks.append(async_save_all_npu_util())
        await asyncio.gather(
            *tasks,
            asyncio.sleep(SLEEP_TIME)
        )

asyncio.run(main())

