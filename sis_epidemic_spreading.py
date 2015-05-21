#!/opt/virtualenv/complex_networks/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
import logging
import random
import pickle
import matplotlib
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
# draw graphs without X11 in linux
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

# set up logging to file
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    #datefmt='%m-%d %H:%M',
                    filename='./epidemic_status.log',
                    filemode='w')
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

class SIS_model(object):
    """
    graph = scale-free, erdos_renyi, etc..
    mu = spontaneous recovery probability
    beta = infection probability of a susceptible (S) individual when it is contacted by an infected (I) one
    p0 = initial fraction of infected nodes. [0-1) value
    seed = random seed
    """
    def __init__(self, name, graph, mu, beta, p0, seed = None):
        self.name = name
        self.graph = graph
        self.mu = mu
        self.beta = beta
        self.p0 = p0

        if seed:
            random.seed(seed)

        self.restart()

    def restart(self):
        """initialize the graph"""
        self.infected_nodes = []
        self.susceptible_nodes = []

        #initialize the graph with Susceptible or Infected individuals
        for node_id in self.graph.nodes():
            is_infected = random.random() < self.p0
            if is_infected:
                self.graph.node[node_id] = 'I'
                self.infected_nodes.append(node_id)
            else:
                self.graph.node[node_id] = 'S'
                self.susceptible_nodes.append(node_id)

    def fraction_of_infected_nodes(self):
        return float(len(self.infected_nodes)) / self.graph.number_of_nodes()

    def simulation_step(self):
        """
        * For each infected node at time step t, we recover it with probability µ: we generate
        a uniform random number between 0.0 and 1.0, and if the value is lower than µ
        the state of that node in the next time step t+1 will be susceptible, otherwise it will
        remain being infected.
        * For each susceptible node at time step t, we traverse all of its neighbors. For each
        infected neighbor (at time step t), the reference node becomes infected with
        probability β. For example, if node A has 6 neighbors, 4 of them being infected, we
        repeat 4 times the generation of a random number and its comparison with β. If at
        the third attempt the random number is lower than β, node A will be infected in
        the next time step t+1, and we may stop the generation of the remaining random
        number; otherwise, node A will continue to be susceptible at time step t+1. Of
        course, the larger the number of infected neighbors, the larger the probability of
        becoming infected.
        """
        future_infected_nodes = []
        future_susceptible_nodes = []

        #we run the recovery process to infected nodes
        for infected_node in self.infected_nodes:
            if random.random() < self.mu:
                #the individual is not infected anymore
                future_susceptible_nodes.append(infected_node)
            else:
                #the individual remain infected
                future_infected_nodes.append(infected_node)

        #we run the infection process to susceptibles nodes
        for susceptible_node in self.susceptible_nodes:
            neighbors = self.graph.neighbors(susceptible_node)
            num_neighbors = len(neighbors)
            i = 0
            is_infected = False
            while i < num_neighbors and not is_infected:
                neighbor = neighbors[i]
                if self.graph.node[neighbor] == 'I':
                    is_infected = random.random() < self.beta
                i += 1

            if is_infected:
                #the individual get infected
                future_infected_nodes.append(susceptible_node)
            else:
                #the individual remain susceptible
                future_susceptible_nodes.append(susceptible_node)

        #update the new states of the nodes
        for future_infected_node in future_infected_nodes:
            self.graph.node[future_infected_node] = 'I'

        for future_susceptible_node in future_susceptible_nodes:
            self.graph.node[future_susceptible_node] = 'S'

        self.infected_nodes = future_infected_nodes
        self.susceptible_nodes = future_susceptible_nodes

class MonteCarloSimlulation(object):
    def __init__(self, model, n_rep, t_max, t_trans):
        self.model = model
        self.n_rep = n_rep
        self.t_max = t_max
        self.t_trans = t_trans

        #debug data
        self.data = []
        self.average = []

    def run_one(self):
        """
        run t_max steps from model and get the data for each one
        """
        simulation_data = []
        for i in xrange(self.t_max):
            #we run one simulation from model and save the fraction in data
            self.model.simulation_step()
            p = self.model.fraction_of_infected_nodes()
            simulation_data.append(p)

        stationary_data = simulation_data[self.t_trans:]

        average_infected_nodes = sum(stationary_data)/len(stationary_data)
        return average_infected_nodes, simulation_data

    def run(self):
        """
        run n_rep complete simulations and get the average of fraction_of_infected_nodes
        """
        full_average = 0
        logging.debug("Starting simulation of %s" % self.model.name)
        for i in xrange(self.n_rep):
            logging.debug("Running simulation %d" % i)
            average_infected_nodes, simulation_data = self.run_one()
            full_average += average_infected_nodes
            if i < self.n_rep - 1:
                #we don't need to restart the graph in the last iteration
                self.model.restart()

            logging.debug("Simulation complete. Fraction of infected nodes = %f" % average_infected_nodes)

            #debug data
            self.data.append(simulation_data)
            self.average.append(average_infected_nodes)

        full_average /= self.n_rep
        logging.debug("Finishing simulation of %s. Full average of infected nodes = %f" % (self.model.name, full_average))
        return full_average

class Exercise(object):
    def __init__(self, use_cache = True):
        self.cache_filename = 'epidemic_cache.pkl'
        self.version = 1.0
        self.use_cache = use_cache
        self.seed = 200
        self.n_rep = 100
        self.t_max = 1000
        self.t_trans = 900
        self.p0 = 0.2

        self.list_num_nodes = [ 500, 1000 ]
        self.mu_values = [ 0.1, 0.5, 0.9 ]

        self.models_cfg = [
            {
                "name": "Barabasi Albert",
                "graph": nx.barabasi_albert_graph,
                "args": { "m": 10 }

            },
            {
                "name": "Erdos Renyi",
                "graph": nx.erdos_renyi_graph,
                "args": { "p": 0.4 }

            },
            {
                "name": "Random network",
                "graph": nx.random_regular_graph,
                "args": { "d": 10 }

            }
        ]

    def load_cache(self):
        data = None
        try:
            with open(self.cache_filename, 'rb') as filecache:
                cache_data = pickle.load(filecache)

            if 'version' in cache_data and cache_data['version'] == self.version:
                data = cache_data['data']
            else:
                logging.error("Error loading cache. Not valid version.")
        except:
            logging.error("Error loading cache")

        return data

    def save_cache(self, data):
        try:
            with open(self.cache_filename, 'wb') as filecache:
                pickle.dump({'version': self.version, 'data': data}, filecache)
        except:
            logging.error("Error saving cache")

    def plot_transitions(self, data_list):
        """
        Receives a data with graph results and draw the transitions with gnuplot
        """

        t = range(1000)
        for data in data_list:
            legend = []
            i = 8
            while i < 51:
                beta = i * 0.02
                legend.append('Beta = %0.2f' % beta)
                p_t = data['simulations'][i]['data'][0]
                i += 10

                plt.plot(t, p_t)

            plt.legend(legend, loc='bottom right')
            plt.xlabel('t')
            plt.ylabel('P')

            args = ['%s = %s' % (arg, data['args'][arg]) for arg in data['args']]
            image_title = "SIS (%s N = %s, %s), (mu = %s, p0 = %s)" % (data['graph_name'], data['num_nodes'], ', '.join(args), data['mu'], data['p0'])

            plt.title(image_title)
            plt.axis([0, 1000, 0, 0.7])

            graph_name_no_spaces = data['graph_name'].replace(' ', '_')
            filename = "images_transitions/SIS_%s_N_%d_%d.png" % (graph_name_no_spaces, data['num_nodes'], i)
            plt.savefig(filename)
            plt.close()

    def process(self, data_list):
        """
        Receives a data with graph results and draw the info with gnuplot
        """

        self.plot_transitions(data_list)

        def float_list_to_string(float_list):
            return [ "%0.2f" %  float_value for float_value in float_list ]

        logging.info("Saving plot images")
        i = 0
        for data in data_list:
            beta_summary = data['beta_list'][:5]
            beta_summary.extend(data['beta_list'][-5:])
            p_summary = data['p_list'][:5]
            p_summary.extend(data['p_list'][-5:])

            beta_summary = float_list_to_string(beta_summary)
            p_summary = float_list_to_string(p_summary)

            logging.info("Drawing model: %s" % data['graph_name'])
            logging.info("Num nodes: %d" % data['num_nodes'])
            logging.info("Mu: %f" % data['mu'])
            logging.info("p0: %f" % data['p0'])
            logging.info("Beta: (%s)" % ','.join(beta_summary))
            logging.info("seed: (%s)" % ','.join(p_summary))

            plt.plot(data['beta_list'], data['p_list'], 'ro')
            plt.xlabel('Beta')
            plt.ylabel('P')

            args = [ '%s = %s' % (arg, data['args'][arg]) for arg in data['args'] ]
            image_title = "SIS (%s N = %s, %s), (mu = %s, p0 = %s)" % (data['graph_name'], data['num_nodes'], ', '.join(args), data['mu'], data['p0'])

            plt.title(image_title)
            plt.axis([0, 1, 0, 1])

            graph_name_no_spaces = data['graph_name'].replace(' ', '_')
            filename = "images/SIS_%s_N_%d_%d.png" % (graph_name_no_spaces, data['num_nodes'], i)
            plt.savefig(filename)
            i += 1
            plt.close()

    def execute(self):
        error_code = 0
        data = None
        save_data = False

        if self.use_cache:
            data = self.load_cache()

        if not data:
            data = self.start()
            save_data = True

        if data:
            if save_data:
                self.save_cache(data)
            self.process(data)
        else:
            logging.error("Fail in simulation")
            error_code = 1

        return error_code

    def start(self):
        list_data = []
        #we run the simulation in all the models
        for model_cfg in self.models_cfg:
            graph_class = model_cfg["graph"]
            graph_name = model_cfg["name"]
            graph_args = model_cfg["args"]

            #we run simulations for several number of nodes
            for num_nodes in self.list_num_nodes:
                graph = graph_class(n = num_nodes, seed = self.seed, **graph_args)

                #export graph in pajek format
                graph_name_no_spaces = graph_name.replace(' ', '_')
                pajek_filename = "net/%s_n_%d.net" % (graph_name_no_spaces, num_nodes)
                nx.write_pajek(graph, pajek_filename)

                #we run simulations for several mu values
                for mu in self.mu_values:
                    #we run simulations for several beta values. Δβ=0.02
                    beta = 0

                    data = {
                        "graph_name": graph_name,
                        "args": graph_args,
                        "seed": self.seed,
                        "p0": self.p0,
                        "num_nodes": num_nodes,
                        "mu": mu
                    }

                    beta_list = []
                    p_list = []

                    data_simulations = []

                    for i in xrange(0, 51):
                        logging.info("Evaluating model: %s" % graph_name)
                        logging.info("Num nodes: %d" % num_nodes)
                        logging.info("Mu: %f" % mu)
                        logging.info("Beta: %f" % beta)
                        logging.info("p0: %f" % self.p0)
                        logging.info("seed: %d" % self.seed)
                        sis_model = SIS_model(graph_name, graph, mu, beta, self.p0, self.seed)
                        simulation = MonteCarloSimlulation(sis_model, self.n_rep, self.t_max, self.t_trans)
                        sim_data = simulation.run()
                        logging.info("Simulation results: %f" % sim_data)

                        #saving data
                        beta_list.append(beta)
                        p_list.append(sim_data)

                        #debug data
                        data_simulations.append({
                            'data': simulation.data,
                            'average': simulation.average
                        })

                        beta += 0.02
                    data['beta_list'] = beta_list
                    data['p_list'] = p_list
                    data['simulations'] = data_simulations

                    list_data.append(data)
        return list_data


if __name__ == '__main__':
    exercise = Exercise()
    exercise.execute()
