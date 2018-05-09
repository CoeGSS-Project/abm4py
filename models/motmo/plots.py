import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from classes_motmo import MPI

sns.set()
sns.set_color_codes("dark")
comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()


def calGreenNeigbourhoodShareDist(earth):
    global PERS
    nPersons = len(earth.getNodeDict(PERS))
    relarsPerNeigborhood = np.zeros([nPersons, 3])
    for i, persId in enumerate(earth.getNodeDict(PERS)):
        person = earth.getEntity(persId)
        x, __ = person.getConnNodeValues('mobType', PERS)
        for mobType in range(3):
            relarsPerNeigborhood[i, mobType] = float(
                np.sum(np.asarray(x) == mobType)) / len(x)

        n, bins, patches = plt.hist(relarsPerNeigborhood, 30, normed=0, histtype='bar',
                                    label=['brown', 'green', 'other'])
        plt.legend()


def incomePerNetwork(earth):
    global PERS
    incomeList = np.zeros([len(earth.nodeDict[PERS]), 1])
    for i, persId in enumerate(earth.nodeDict[PERS]):
        person = earth.entDict[persId]
        x, friends = person.getConnNodeValues('mobType', PERS)
        incomes = [earth.entDict[friend].hh.getValue('income') for friend in friends]
        incomeList[i, 0] = np.mean(incomes)

    n, bins, patches = plt.hist(incomeList, 20, normed=0, histtype='bar',
                                label=['average imcome '])
    plt.legend()


def computingTimes(earth):
    plt.figure()

    allTime = np.zeros(earth.nSteps)
    colorPal = sns.color_palette("Set2", n_colors=5, desat=.8)

    for i, var in enumerate([earth.computeTime, earth.waitTime, earth.syncTime, earth.ioTime]):
        plt.bar(np.arange(earth.nSteps), var, bottom=allTime, color=colorPal[i], width=1)
        allTime += var
    plt.legend(['compute time', 'wait time', 'sync time', 'I/O time'])
    plt.tight_layout()
    plt.ylim([0, np.percentile(allTime, 99)])
    plt.savefig(earth.para['outPath'] + '/' + str(mpiRank) + 'times.png')
