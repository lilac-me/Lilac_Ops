RANK_SIZE = 8
LOG_FILE = "npu_util.log"
with open(LOG_FILE, "r") as f:
    logs = f.readlines()

total = 0
sum = 0
for log in logs:
    # 2025-05-30 17:43:51.872844, rank=5, util=37
    try:
        util = int(log[log.find("util=")+5:])
        total += util
        sum += 1
    except Exception as e:
        print(e)

print(total, sum, total/sum)
