import data_manager as data_m
import parameters as params
import initialize
import evolver 

# To use a profiler
# python3.11 -m cProfile main.py

# running the simulation through the Evolve Class
# alarms = True in execute
evolver.Evolve(params.version, initialize.initial_conditions(params.problem)).execute(params.alarms)

