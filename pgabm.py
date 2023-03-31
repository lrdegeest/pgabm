import pandas as pd
import numpy as np
import random as random
import string as string
import tqdm as tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = ['monospace']


# ==========================================================================
# SIMULATION CLASS
# ==========================================================================

class simulation(object):

    def __init__(
            self,
            alpha=0.4,
            n_agents=100,
            groupsize=5,
            n_generations=25,
            n_trials=100,
            n_rounds=20,
            evolve_penalty=True,
            penalty=None,
            evolve_threshold=True,
            threshold=None,
            social_choice_mechanism='median',
            ROT=True,
            optimizeROT=False,
            optimizeContribution=False,
            mutation_rate=0.33,
            record_only_last_gen=False,
            hardcode_cooperators=False,
            hardcode_institution=False,
            n_cooperators=None):
        self.n_agents = int(n_agents)
        self.groupsize = groupsize
        self.n_groups = int(self.n_agents / self.groupsize)
        self.n_generations = int(n_generations)
        self.n_trials = int(n_trials)
        self.alpha = alpha
        self.evolve_penalty = evolve_penalty
        self.penalty = penalty
        self.evolve_threshold = evolve_threshold
        self.threshold = threshold
        self.social_choice_mechanism = social_choice_mechanism.lower()
        self.mutation_rate = mutation_rate
        self.data_all_trials = []
        self.n_rounds = n_rounds
        self.ROT = ROT
        self.optimizeROT = optimizeROT
        self.optimizeContribution = optimizeContribution
        self.record_only_last_gen = record_only_last_gen
        self.hardcode_cooperators = hardcode_cooperators
        self.hardcode_institution = hardcode_institution
        self.n_cooperators = n_cooperators

    def instantiate_agents(self):
        # agent: [contribution (0), threshold (1), penalty (2),   ROT param (3), payoff (4)]
        # self.agents = [[random.randint(0,100) ,random.randint(0,100),random.randint(0,100), random.random(), 0.0] for i in range(self.n_agents)]
        self.agents = [
            [random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), 0.0] for i
            in range(self.n_agents)]
        ## note: to test system at T=100 change *the first* random.choice(range(0,101)) to (0,111)

        ## this next part is for testing the system at T=100
        ### agents whose contribution is above 100: force the contribution to 100 and push the remainder to their payoffs
        ### uncomment to run
        # for agent in self.agents:
        #    if agent[0] > 100: # if they have contribution > 100
        #        agent[4] += 100 - agent[0] # push the remainder to their payoffs
        #        agent[0] = 100 # reset their contribution to zero
        # switches determine whether you assign a parameter or allow it to evolve
        # case 1: threshold and penalty are both exogenously assigned
        if not (self.evolve_penalty and self.evolve_threshold):
            for agent in self.agents:
                agent[1] = self.threshold
                agent[2] = self.penalty
        # case 2
        elif (self.evolve_penalty == False) and (self.evolve_threshold == True):
            for agent in self.agents:
                agent[2] = self.penalty
                # case 3
        elif (self.evolve_penalty == True) and (self.evolve_threshold == False):
            for agent in self.agents:
                agent[1] = self.threshold
        # hardcode cooperators, only using this for 3-way co-evolution experiments
        elif self.hardcode_cooperators:
            for i in range(self.n_cooperators):
                self.agents[i][0] = 100
        elif self.hardcode_institution:
            for i in range(self.n_cooperators):
                self.agents[i][1:2] = 100, 100
        pass

    @staticmethod
    def midrange_mean(x):
        # average of ONLY the max and min values of a group's list
        sorted_x = sorted(x)
        m = np.mean([sorted_x[0], sorted_x[len(sorted_x) - 1]])
        return m

    @staticmethod
    def truncated_mean(x):
        # average of EVERYTHING BUT the max and min values of a group's list
        sorted_x = sorted(x)
        m = np.mean(sorted_x[1:len(sorted_x) - 1])
        return m

    def dictator(self):
        picks = [random.choice([agent for agent in self.agents][i:i + self.groupsize]) for i in
                 range(0, self.n_agents, self.groupsize)]
        return picks

    def set_scm(self):
        if self.social_choice_mechanism == 'median':
            self.scm = np.median
        elif self.social_choice_mechanism == 'mean':
            self.scm = np.mean
        elif self.social_choice_mechanism == 'truncated':
            self.scm = self.truncated_mean
        elif self.social_choice_mechanism == 'midrange':
            self.scm = self.midrange_mean
        elif self.social_choice_mechanism == 'dictator':  # JM dictator dictates the T and P params; I need to re-do this!!!
            self.scm = random.choice
        elif self.social_choice_mechanism == 'max':
            self.scm = np.max
        elif self.social_choice_mechanism == 'min':
            self.scm = np.min
        else:
            raise ValueError(
                "social_choice_mechanism must be one of 'mean', 'median', 'dictator', 'min', 'max', 'truncated', or 'midrange'")

    def get_thresholds(self):
        thresholds = [self.scm([agent[1] for agent in self.agents][i:i + self.groupsize]) for i in
                      range(0, self.n_agents, self.groupsize)]
        return thresholds

    def get_penalties(self):
        penalties = [self.scm([agent[2] for agent in self.agents][i:i + self.groupsize]) for i in
                     range(0, self.n_agents, self.groupsize)]
        return penalties

    def get_institution(self):
        if self.scm == "dictator":
            dictators = self.dictator()
            thresholds = [agent[1] for agent in dictators]
            penalties = [agent[2] for agent in dictators]
        else:
            thresholds = [self.scm([agent[1] for agent in self.agents][i:i + self.groupsize]) for i in
                          range(0, self.n_agents, self.groupsize)]
            penalties = [self.scm([agent[2] for agent in self.agents][i:i + self.groupsize]) for i in
                         range(0, self.n_agents, self.groupsize)]
        return thresholds, penalties
        pass

    def play_game(self):
        for i in range(self.n_rounds):
            # 1. shuffle the agents into groups
            random.shuffle(self.agents, random.random)
            # 2. get T and P for each group
            ## every scenario has a list of thresholds and penalities and both lists are length n_groups
            if self.evolve_threshold and self.evolve_penalty:  # evolving institutions
                # thresholds and penalties are determined by a fixed social choice mechanism
                thresholds, penalties = self.get_institution()
            else:
                if self.optimizeROT:
                    thresholds = [random.randint(0, 100) for i in range(self.n_groups)]
                    penalties = [random.randint(0, 100) for i in range(self.n_groups)]
                else:
                    thresholds = [self.threshold] * self.n_groups
                    penalties = [self.penalty] * self.n_groups
            # 3. get individual contributions
            for j in range(self.n_groups):
                for agent in self.agents[j * self.groupsize:j * self.groupsize + self.groupsize]:
                    if self.ROT:
                        agent[0] = thresholds[j] if (agent[3] / 100.0) * thresholds[j] < penalties[j] else 0
                    elif self.optimizeContribution:
                        agent[0] = thresholds[j] if (thresholds[j] * (1.0 - self.alpha) < penalties[j]) else 0
                    else:
                        pass  # use agent[0] from instantiation or evolution
            # 4. sum up contributions in each group
            group_contributions = [sum([agent[0] for agent in self.agents][i:i + self.groupsize]) for i in
                                   range(0, self.n_agents, self.groupsize)]
            # 5. calculate payoffs
            for j in range(self.n_groups):
                for agent in self.agents[j * self.groupsize:j * self.groupsize + self.groupsize]:
                    # payoff from the public good
                    agent[4] += (100.0 - agent[0]) + self.alpha * group_contributions[j]  # initial payoff
                    # a test of the adaptive algorithm, see email from JM 4/30/21. uncomment to run (and comment out payoffs above)
                    # agent[4] += 0 - abs(agent[1] - 0) - abs(agent[2] - 86) - abs(agent[3] - 100)
                    # adjust payoff with penalty if agent is noncompliant
                    if agent[0] < thresholds[j]: agent[4] -= penalties[j]  # pay the penalty if c < T
        pass  # END

    def evolve(self):
        # new generation of agents
        self.new_agents = []
        # 1. tournament selection; winner is the parent
        while len(self.new_agents) < self.n_agents:
            two_agents = random.sample(self.agents, 2)
            if two_agents[0][4] > two_agents[1][4]:  # compare payoffs
                parent = two_agents[0]
            else:
                parent = two_agents[1]
            child = [0.0] * 5
            child[0:4] = parent[0:4]  # copy the parent to the child except the payoffs
            self.new_agents.append(child)
        # 2. mutate
        for new_agent in self.new_agents:
            p = random.random()
            if p <= self.mutation_rate:
                which_attribute = random.randint(1, 3)  # could be 1 (threshold), 2 (penalty) or 3 (ROT)
                new_agent[which_attribute] += 5 - random.randint(0,
                                                                 5 * 2)  # updated to match JM's code. mutation is [-5, ... , 5] i.e. 5% mutation (abs(5)/100)
                if new_agent[which_attribute] > 100: new_agent[which_attribute] = 100
                if new_agent[which_attribute] < 0: new_agent[which_attribute] = 0
                # 3. update the population
        self.agents = self.new_agents

    def trial(self):  # run N generations
        self.instantiate_agents()  # each trial starts with fresh agents
        self.set_scm()
        self.data_trial_list = []
        for i in range(self.n_generations):  # for each generation
            self.play_game()  # play game G_g for n_rounds
            ## record data: add G_g data for given trial to list; this list is created by run()
            if self.record_only_last_gen == True:
                if i == (self.n_generations - 1):
                    self.data_trial_list.append(self.agents)
            else:
                self.data_trial_list.append(self.agents)
            self.evolve()
        pass

    def print_settings(self):
        print("Social choice mechanism is " + str(self.social_choice_mechanism).upper())
        print("ROT is " + str(self.ROT).upper() + " and optimizeROT is " + str(self.optimizeROT).upper())
        print("evolve penalty is " + str(self.evolve_penalty).upper())
        print("evolve threshold is " + str(self.evolve_threshold).upper())
        pass

    def run(self):  # run N trials
        self.print_settings()
        with tqdm.tqdm_notebook(total=self.n_trials) as progressbar:
            for i in range(self.n_trials):
                progressbar.set_description('TRIALS: %d' % (i + 1))
                # self.data_trial_list = [] # you have to empty this list at the start of each trial
                self.trial()  # returns list of generation data for trial i (data_trial_list)
                self.data_all_trials.append(self.data_trial_list)  # adds the list of data from a trial
                progressbar.update()
            self.data = np.array(self.data_all_trials, dtype="float64")
        pass

    def run_no_progressbar(self):
        # run N trials
        for i in range(self.n_trials):
            self.data_trial_list = []  # you have to empty this list at the start of each trial
            self.trial()  # returns list of generation data for trial i (data_trial_list)
            self.data_all_trials.append(self.data_trial_list)  # adds the list of data from a trial
        self.data = np.array(self.data_all_trials, dtype="float64")
        pass


# ==========================================================================
# EXPERIMENT CLASS
# ==========================================================================

class experiment(object):

    def __init__(
            self,
            n_agents=100,
            n_generations=25,
            n_trials=10,
            n_rounds=20,
            evolve_penalty=True,
            penalties=None,
            evolve_threshold=True,
            thresholds=None,
            social_choice_mechanism='median',
            ROT=False,
            optimizeROT=False):
        self.n_agents = n_agents
        self.n_generations = n_generations
        self.n_trials = n_trials
        self.sim_data_list = []  # why do i need this? i think it was to store all the raw data from sims
        self.evolve_penalty = evolve_penalty
        self.evolve_threshold = evolve_threshold
        self.penalties = [penalties] if type(penalties) is not list else penalties
        self.thresholds = [thresholds] if type(thresholds) is not list else thresholds
        self.n_thresholds = len(self.thresholds)
        self.n_penalties = len(self.penalties)
        self.n_rounds = n_rounds
        self.ROT = ROT
        self.optimizeROT = optimizeROT
        pass

    def run_experiment_fixed_institution(self):
        self.data = []
        if self.ROT == False:  # absolute contributions
            with tqdm.tqdm_notebook(total=self.n_thresholds) as pbar1:
                for i in self.thresholds:
                    pbar1.set_description('threshold: %d' % (i))
                    self.avg_contributions = []  # column of threshold i
                    for j in self.penalties:
                        s = simulation(
                            n_agents=self.n_agents,
                            n_generations=self.n_generations,
                            n_trials=self.n_trials,
                            n_rounds=self.n_rounds,
                            evolve_penalty=self.evolve_penalty,
                            penalty=j,
                            evolve_threshold=self.evolve_threshold,
                            threshold=i)
                        s.run_no_progressbar()
                        # now you have data: s.data, a numpy array of all trials for penalty j and threshold i
                        # get the average contribution for i,j
                        ## note: s.data is indexed as so: s.data[trials, gens, agents, attributes]
                        ## so s.data[:,self.n_generations - 1,:,0] means "all trials, last generation, all agents, first attribute (contribution)"
                        avg_contribution = np.mean(s.data[:, self.n_generations - 1, :, 0].flatten())
                        self.avg_contributions.append(avg_contribution)  # add i,j mean outcome to list
                    self.data.append(self.avg_contributions)
                    pbar1.update()
            self.results = pd.DataFrame(self.data).T.round(
                decimals=0) / 100  # threshold columns, penalty rows, round and then divide by 100 to view contributions as percent of optimal
            self.results.index = ['P' + str(i) for i in self.penalties]
            self.results.columns = ['T' + str(i) for i in self.thresholds]
        else:
            with tqdm.tqdm_notebook(total=self.n_thresholds) as pbar1:
                for i in self.thresholds:
                    pbar1.set_description('threshold: %d' % (i))
                    self.avg_contributions = []  # column of threshold i
                    for j in self.penalties:
                        s = simulation(
                            n_agents=self.n_agents,
                            n_generations=self.n_generations,
                            n_trials=self.n_trials,
                            n_rounds=self.n_rounds,
                            evolve_penalty=self.evolve_penalty,
                            penalty=j,
                            evolve_threshold=self.evolve_threshold,
                            threshold=i,
                            ROT=True)
                        s.run_no_progressbar()
                        # now you have data: s.data, a numpy array of all trials for penalty j and threshold i
                        # get the average contribution for i,j
                        ## note: s.data is indexed as follow: s.data[trials, gens, agents, attributes]
                        ## so s.data[:,self.n_generations - 1,:,0:1] means "all trials, last generation, all agents, first attribute (contribution)"
                        ## here i'm collecting average param (the ROT) but i don't want to change list names
                        avg_contribution = np.mean(s.data[:, self.n_generations - 1, :, 3].flatten())
                        self.avg_contributions.append(avg_contribution)  # add i,j mean outcome to list
                    self.data.append(self.avg_contributions)
                    pbar1.update()
            self.results = pd.DataFrame(self.data).T.round(
                decimals=2)  # threshold columns, penalty rows, round to 2 decimals
            self.results.index = ['P' + str(i) for i in self.penalties]
            self.results.columns = ['T' + str(i) for i in self.thresholds]
        pass

    # ==========================================================================


# UTILS
## for system 3a'
# ==========================================================================

def run_sims(n_trials=100,
             n_generations=25,
             groupsize=5,
             alpha=0.4,
             ROT=True,
             scm=['median', 'mean', 'dictator', 'max', 'min', 'truncated', 'midrange']):
    # first run all the sims
    # all_sims is an array: (scm, n_trials, n_gens, n_agents, attributes)
    all_sims = []
    with tqdm.tqdm_notebook(total=len(scm)) as progressbar:
        for i in scm:
            progressbar.set_description('SCM = %i' % (scm.index(i) + 1))
            sim = simulation(n_trials=n_trials,
                             n_generations=n_generations,
                             ROT=ROT,
                             alpha=alpha,
                             groupsize=groupsize,
                             social_choice_mechanism=i)
            sim.run_no_progressbar()
            all_sims.append(sim.data)
            progressbar.update()
    return (all_sims)


def convert_to_df(array, gen, scm=['median', 'mean', 'dictator', 'max', 'min', 'truncated', 'midrange']):
    # takes the numpy array from run_3body and returns averages by trial x scm for a given generation
    # not great because the scm arg HAS to be in the same order as the they were inputted for run_sims()
    # but whatever
    list_of_dfs = []
    for i in scm:
        scm_index = scm.index(i)
        # MEAN
        ## mean of each attribute in each trial, then turn into a dataframe
        df_mean = pd.DataFrame(np.mean(array[scm_index][:, gen, :, :], axis=1),
                               columns=["contribution", "threshold", "penalty", "rot", "payoff"])
        ## calculate expected contributions based on average values and heuristic
        df_mean['expected_contribution'] = np.where((1 - df_mean['rot']) * df_mean['threshold'] < df_mean['penalty'],
                                                    df_mean['threshold'], 0)
        ## add the stat type
        df_mean['stat'] = 'mean'
        # SD
        df_sd = pd.DataFrame(np.std(array[scm_index][:, gen, :, :], axis=1),
                             columns=["contribution", "threshold", "penalty", "rot", "payoff"])
        ## add the stae stype
        df_sd['stat'] = 'sd'
        # JOIN
        df = pd.concat([df_mean, df_sd])
        ## add the scm column
        df['scm'] = i.upper()
        ## add the trial; it's just the length of the df
        df['trial'] = np.arange(len(df))
        ## renaming for nicer plots
        df['scm_plot'] = df['scm'].replace({'DICTATOR': 'DICTATOR\n(random)',
                                            'MAX': 'DICTATOR\n(max)',
                                            'MIN': 'DICTATOR\n(min)',
                                            'MEDIAN': 'TEAM\n(median)',
                                            'MEAN': 'TEAM\n(mean)',
                                            'MEAN': 'TEAM\n(mean)',
                                            'TRUNCATED': 'TEAM\n(truncated)',
                                            'MIDRANGE': 'TEAM\n(midrange)'})
        list_of_dfs.append(df)
    dfs = pd.concat(list_of_dfs)
    return (dfs)


def get_all_gens(data, max_gen=25, summarize=False):
    l = []
    for i in range(max_gen):
        gendf = convert_to_df(data, gen=i)
        if summarize:
            gendf = gendf.groupby('scm').mean().reset_index()
        else:
            pass
        gendf['gen'] = i
        l.append(gendf)
    all_gens_df = pd.concat(l)
    return (all_gens_df)


def summarize(data, stat, gen):
    gendf = convert_to_df(data, gen=gen)
    gendf = gendf[gendf['stat'] == stat.lower()]
    gendf = gendf.groupby('scm').mean().reset_index()
    return (gendf)


def rank_scms(data, column, yes_print=True, title=""):
    # data is the data from pgabm.run_sims()
    # column is a string column name
    data = data[data['stat'] == 'mean']
    data['99_contribution'] = np.where(data[column] >= 99.5, 1, 0)
    data['95_contribution'] = np.where(data[column] >= 95.0, 1, 0)
    data = data[['scm', column, '99_contribution', '95_contribution']].groupby('scm').mean().reset_index()
    if yes_print:
        print("Ranked by " + title + " contributions\n")
        data = data[['scm', column, '99_contribution', '95_contribution']]
        print(data.sort_values(by=[column], ascending=False).to_string(index=False, header=True))
    else:
        return data


def plot(data, attributes, stat, gen=0, ymin=0, ymax=105, evolution=False, max_gen=24, title=""):
    """
    plot(attributes = ['threshold', 'penalty'], gen = 24, title = "Generation 24")
    plot(attributes = 'expected_contribution', gen = 0, title = "Generation 0")
    plot(attributes = 'expected_contribution', evolution = True)
    """
    attributes = [attributes] if isinstance(attributes, str) else attributes
    if evolution:
        l = []
        for i in range(max_gen):
            gendf = convert_to_df(data, gen=i)
            gendf = gendf[gendf['stat'] == stat.lower()]
            gendf = gendf.groupby('scm_plot').mean().reset_index()
            gendf['gen'] = i
            l.append(gendf)
        all_gens_df = pd.concat(l)
        hue_order = sorted(all_gens_df['scm_plot'].unique().tolist(), reverse=True)
        fig, ax = plt.subplots(ncols=len(attributes), figsize=(8 * len(attributes) + 1, 6), squeeze=False)
        for a in attributes:
            i = attributes.index(a)
            sns.lineplot(x="gen", y=a, hue="scm_plot", hue_order=hue_order, data=all_gens_df, ax=ax[0, i])
            ax[0, i].set_xlabel("Generation")
            ax[0, i].set_ylabel("")
            ax[0, i].set_title(a.upper().replace("_", " "), fontsize=18)
            ax[0, i].set_ylim([ymin, ymax])
            ax[0, i].get_legend().set_title("")
            ax[0, i].legend(loc='best', ncol=2)
        fig.suptitle(title, fontsize=25)
    else:
        ## plot the distributions with violin plots
        gen_data = convert_to_df(data, gen=gen)
        gen_data = gen_data[gen_data['stat'] == stat.lower()]
        hue_order = sorted(gen_data['scm_plot'].unique().tolist(), reverse=True)
        fig, ax = plt.subplots(ncols=len(attributes), figsize=(8 * len(attributes) + 1, 6), squeeze=False)
        for a in attributes:
            i = attributes.index(a)
            sns.violinplot(x="scm", y=a, data=gen_data, cut=0, ax=ax[0, i])
            ax[0, i].set_xlabel("")
            ax[0, i].set_ylabel("")
            ax[0, i].set_ylim([ymin, ymax])
            ax[0, i].set_title(a.upper().replace("_", " "), fontsize=18)
        fig.suptitle(title, fontsize=25)
    # show whichvever plot you made
    plt.show()


# ==========================================================================
# UTILS
## utils for running a monte carlo on each institution
# ==========================================================================

class MonteCarloSCM(object):
    
    def __init__(self, groupsize = 5, niter = 1000):
        self.groupsize = groupsize
        self.scms = [np.mean, truncated_mean, midrange_mean,  np.median, np.max, np.min, np.random.choice]
        self.names = ["mean", "truncated", "midrange", "median", "max", "min", "random"]
        self.niter = niter
        self.sim_results = pd.DataFrame()
    
    @staticmethod
    def midrange_mean(x):
        # average of only the max and min values of a group's list
        extremes = [np.min(x), np.max(x)]
        trimmed_x = [i for i in x if i in extremes]
        return np.mean(trimmed_x)

    @staticmethod
    def truncated_mean(x):
        # average of everything but the max and min values
        extremes = [np.min(x), np.max(x)]
        trimmed_x = [i for i in x if i not in extremes]
        return np.mean(trimmed_x)
    
    def run(self):
        pds = []
        for scm, j in zip(self.scms, self.names):
            thresholds, penalties = [], []
            d = pd.DataFrame()
            for i in range(self.niter):
                t = random.sample(range(0,101), self.groupsize)
                p = random.sample(range(0,101), self.groupsize)
                choice_t = scm(t)
                thresholds.append(choice_t)
                choice_p = scm(p)
                penalties.append(choice_p)
            d['threshold'] = thresholds
            d['penalty'] = penalties
            d['scm'] = j.upper()
            pds.append(d)
        pd_return = pd.concat(pds)
        pd_return = pd_return.reset_index(drop=True)
        self.sim_results = pd_return
        pass
    
    def plot(self):
        sns.displot(data = self.sim_results, x="threshold", y = 'penalty', col="scm", col_wrap = 3, kind="kde", fill=True)
        plt.show()
        pass
    
    def summary(self):
        dfs = self.sim_results.groupby('scm').mean()
        return dfs


# ==========================================================================
# UTILS
## find nash equilibria
# ==========================================================================
class NashEquilibria(object):

    def __init__(self, alpha, threshold, penalty, n=5, levels=3, return_numeric=False):
        self.pi = None
        self.best_response_letters = None
        self.best_response_positions = None
        self.best_response = None
        self.l = ''
        self.alpha = alpha
        self.n = n
        self.L = levels
        self.T = threshold
        self.P = penalty
        self.return_numeric = return_numeric

    def instantiate(self):
        self.choices = [i / (self.L - 1) for i in range(self.L)]
        if len(self.choices) > len(list(string.ascii_lowercase)):
            raise ValueError("Choose an L smaller than 26")
        else:
            self.letters = list(string.ascii_lowercase)[0:len(self.choices)]
        pass

    def get_cstar(self):
        a = [c >= self.threshold for c in self.choices]
        if (1 - self.alpha) * self.threshold > self.penalty:
            # optimal contribution is 0
            self.cstar = 0
        else:
            # optimal contribution is T
            # first find minimal c that satisfies threshold
            for c in range(len(a)):
                if a[c]:
                    self.cstar = self.choices[c]
                    break
        pass

    def get_pi(self, ci):
        self.pi = (1 - ci) + self.alpha * (ci + (self.n - 1) * self.cstar)
        if ci < self.T:
            self.pi = self.pi - self.P
        else:
            pass
        self.pi = round(self.pi, 3)
        return self.pi

    def get_best_response(self):

        # first calculate payoffs for all choices when agents j \neq i plays cstar
        all_pi = [0] * len(self.choices)
        for i in range(len(self.choices)):
            all_pi[i] = self.get_pi(ci=self.choices[i])
            pass

        # next figure out which payoffs are highest
        is_max_pi = [i == max(all_pi) for i in all_pi]

        # finally, get the strategies that correspond to max payoffs
        self.best_response = []
        self.best_response_positions = []
        for i in range(len(is_max_pi)):
            if is_max_pi[i]:
                self.best_response.append(self.choices[i])
                self.best_response_positions.append(i)

        # this is just to get the results in letter format
        self.best_response_letters = []
        for i in self.best_response_positions:
            self.best_response_letters.append(self.letters[i])

        for i in range(len(self.best_response_letters)):
            self.l += self.best_response_letters[i]
        pass

    def run(self):
        self.instantiate()
        self.get_cstar()
        self.get_best_response()
        if self.return_numeric:
            return self.best_response
        else:
            return self.l


def find_nash_equilibria(alpha, n_levels=10, levels=3, return_numeric=False, loud=True):
    thresholds = penalties = [i / n_levels for i in range(n_levels + 1)]
    results = []
    for i in range(len(thresholds)):
        res = [''] * (n_levels + 1)
        for j in range(len(penalties)):
            res[j] = NashEquilibria(alpha=alpha,
                                    levels=levels,
                                    return_numeric=return_numeric,
                                    threshold=thresholds[i],
                                    penalty=penalties[j]).run()
        results.append(res)
    res_df = pd.DataFrame(results)
    if loud:
        print("Nash equilibria for alpha =", alpha, 'and L =', levels)
    return res_df
