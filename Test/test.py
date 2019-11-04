import fileinput
import random
import time
import pandas as pd


def readJobs(path=None):
    """
    Returns a problem instance specified in a textfile at path.
    """
    # abz5 = './abz5.txt'
    # with fileinput.input(files=abz5) as f:
    #     next(f)
    #     jobs = [[(int(machine), int(time)) for machine, time in zip(*[iter(line.split())] * 2)]
    #             for line in f if line.strip()]   # 去除首尾空格
    #     print('jobs:',jobs)

    pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0])
    ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0])

    num_mc = pt_tmp.shape[1]
    num_job = pt_tmp.shape[0]

    jobs = []

    # 机器顺序
    ms = [list(map(int, ms_tmp.iloc[i])) for i in range(num_job)]

    # 加工时间
    pt = [list(map(int, pt_tmp.iloc[i])) for i in range(num_job)]

    for i in range(len(ms)):  # 循环10次 ， 生成10个子列表
        temp1 = []
        for j in range(len(ms[i])):
            temp1.append((ms[i][j], pt[i][j]))
        jobs.append(temp1)
    print('jobs:  ',jobs)
    return jobs


def printJobs(jobs):
    """Print a problem instance."""
    m = len(jobs[0])  # number of machines
    j = len(jobs)   # number of jobs
    print("Number of machines:", m)
    print("Number of jobs:", j)
    curjob = 1 ;
    for job in jobs:
        print('job',curjob,end=':  ')
        curjob += 1
        for machine, time in job:
            print(machine, time, end=", ")
        print()


def cost(jobs, schedule):
    """Calculate the cost of a schedule for a problem instance jobs."""
    j = len(jobs)
    m = len(jobs[0])
    tj = [0] * j  # end of previous task for each job
    tm = [0] * (m+1)  # end of previous task on each machine
    ij = [0] * j  # task to schedule next for each job
    for i in schedule:
        machine, time = jobs[i][ij[i]]   # 第 i 个工序的第 ij[i]个步骤
        ij[i] += 1
        # TODO The estimation of the start time is very rough.
        # A better (?) but slower approach would be to look just for the end
        # of the previous task of the job and search a free slot in the
        # timetable for the machine.
        start = max(tj[i], tm[machine])
        end = start + time
        tj[i] = end
        tm[machine] = end
    return max(tm)


class OutOfTime(Exception):
    pass


def randomSchedule(j, m):
    """
    Returns a random schedule for j jobs and m machines,
    i.e. a permutation of 0^m 1^m ... (j-1)^m = (012...(j-1))^m.
    """
    schedule = [i for i in list(range(j)) for _ in range(m)]
    # schedule: [0,0,0,1,1,1,2,2,2]
    random.shuffle(schedule)  # 随机排序序列
    return schedule


def randomSearch(jobs, maxTime=None):
    """
    Perform random search for problem instance jobs.
    Set maxTime to limit the computation time or raise
    a KeyboardInterrupt (Ctrl+C) to stop.
    """
    numExperiments = 100  # experiments performed per loop
    solutions = []  # list of (time, schedule) with decreasing time
    best = 10000000  # TODO set initial value for max or add check for None in loop
    t0 = time.time()
    totalExperiments = 0
    j = len(jobs)
    m = len(jobs[0])
    rs = randomSchedule(j, m)
    while True:
        try:
            start = time.time()
            for i in range(numExperiments):
                rs = randomSchedule(j, m)
                c = cost(jobs, rs)
                if c < best:    #如果小于当前最优值，则加入到列表中
                    best = c
                    solutions.append((c, rs))
            totalExperiments += numExperiments
            if maxTime and time.time() - t0 > maxTime:
                raise OutOfTime("Time is over")
            t = time.time() - start
            if t > 0:
                print("Best:", best, "({:.1f} Experiments/s, {:.1f} s)".format(
                    numExperiments / t, time.time() - t0))
            # Make outputs appear about every 3 seconds.
            if t > 4:
                numExperiments //= 2
            elif t < 1.5:
                numExperiments *= 2

        except (KeyboardInterrupt, OutOfTime) as e:
            print()
            print("================================================")
            print("Best time:", best, "  (lower bound {})".format(lowerBound(jobs)))
            print("Best solution:")
            print(solutions[-1])
            print("Found in {:} experiments in {:.1f}s".format(totalExperiments, time.time() - t0))
            print()
            # printSchedule(jobs, solutions[-1][1])
            return solutions[-1]


def printSchedule(jobs, schedule):
    # TODO code duplication with cost()
    j = len(jobs)
    m = len(jobs[0])
    tj = [0] * j  # end of previous task for job
    tm = [0] * (m+1)  # end of previous task on machine
    ij = [0] * j  # task to schedule next for each job
    for i in schedule:
        machine, time = jobs[i][ij[i]]
        ij[i] += 1
        start = max(tj[i], tm[machine])
        end = start + time
        tj[i] = end
        tm[machine] = end
        print("Start job {} on machine {} at {} ending {}.".format(i, machine, start, end))
    print("Total time:", max(tm))


def prettyPrintSchedule(jobs, schedule):
    # TODO code duplication with cost
    # TODO Generate an image where each job has a different color
    #      and a timestep is a pixel. This way even schedules with
    #      time ~1000 have a useful representation.
    def format_job(time, jobnr):
        if time == 1:
            return '#'
        if time == 2:
            return '[]'
        js = str(jobnr)
        # TODO number should be repeated for long times
        # but these may not be nice to print anyways...
        # if 2 + len(js) <= time and time < 10:
        if 2 + len(js) <= time:
            return ('[{:^' + str(time - 2) + '}]').format(jobnr)

        return '#' * time

    j = len(jobs)
    m = len(jobs[0])
    tj = [0] * j  # end of previous task for job
    tm = [0] * (m+1)  # end of previous task on machine
    ij = [0] * j  # task to schedule next for each job
    output = [""] * (m+1)
    for i in schedule:
        machine, time = jobs[i][ij[i]]
        ij[i] += 1
        start = max(tj[i], tm[machine])
        space = start - tm[machine]
        end = start + time
        tj[i] = end
        tm[machine] = end
        output[machine] += ' ' * space + format_job(time, i)
    [print(machine_schedule) for machine_schedule in output]
    print("Total Time: ", max(tm))


def numMachines(jobs):
    return len(jobs)


def numJobs(jobs):
    return len(jobs[0])


def lowerBound(jobs):  # 最小的边界
    """Returns a lower bound for the problem instance jobs."""

    # upper boulnd: 1231, lower bound 1005
    def lower0():
        # max min time of jobs
        # each job has to be executed sequentially
        return max(sum(time for _, time in job) for job in jobs)  # 每个工序的最长时间

    def lower1():
        # max min time of machines
        # a machine must process all its tasks
        print(max(numJobs(jobs) ,numMachines(jobs)))
        mtimes = [0] * (max(numJobs(jobs) ,numMachines(jobs))+1)
        # print(mtimes): [0,0,0,0]
        for job in jobs:
            for machine, time in job:
                mtimes[machine] += time
        return max(mtimes)

    return max(lower0(), lower1())


if __name__ == '__main__':
    abz5 = 'abz5.txt'
    jobs = readJobs(abz5)
    j = numJobs(jobs)
    m = numMachines(jobs[0])
    schedule = randomSchedule(j, m)
    printJobs(jobs)
    # printSchedule(jobs, schedule)
    # prettyPrintSchedule(jobs, schedule)
    lowerBound(jobs)
    cost1, solution = randomSearch(jobs, 10)
    print(cost1)
    prettyPrintSchedule(jobs, solution)
    print()
    printSchedule(jobs, solution)

