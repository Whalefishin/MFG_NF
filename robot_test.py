import klampt
from klampt.plan import cspace,robotplanning
from klampt.plan.robotcspace import RobotCSpace
from klampt.io import resource
from klampt.model import collide
import time
from klampt.model.trajectory import RobotTrajectory
import numpy as np
from klampt import vis
import os

### world & robot setup ###
world = klampt.WorldModel()
world.readFile("./Klampt-examples/data/tx90cuptable_2.xml")
robot = world.robot(0)
robot2 = world.robot(0)

# this is the CSpace that will be used.  Standard collision and joint limit constraints
# will be checked
# space = robotplanning.makeSpace(world,robot,edgeCheckResolution=0.05)
space = RobotCSpace(robot, collide.WorldCollider(world))

#fire up a visual editor to get some start and goal configurations
qstart = resource.get("under_table_start_4.config")
# qgoal  = resource.get("cup_end.config")
qgoal  = resource.get("cup_end_2.config")



### path-planning ###
settings = {'type':'rrt',
    'perturbationRadius':0.25,
    'bidirectional':True,
    'shortcut':True,
    'restart':True,
    'restartTermCond':"{foundSolution:1,maxIters:1000}"
}
t0 = time.time()
print("Creating planner...")
#Manual construction of planner
planner = cspace.MotionPlan(space, **settings)
planner.setEndpoints(qstart, qgoal)



# check some path and see if they are feasible
exp_name = 'sigmoid_under_table_3_var=1e-4_lbdF=1e3_lbdOT=1e-1_interp=20_twoInit'
traj_density_path = os.path.join('./results/robot_1', exp_name, 'plot_data/traj_density.npy')
# traj_robot_path   = os.path.join('./results/robot_1', exp_name, 'plot_data/traj_robot.npy')
# traj_robot_path   = os.path.join('./results/robot_1', exp_name, 'plot_data/traj_robot_interp.npy')

### testing
traj_robot_path   = os.path.join('./results/robot_1', exp_name, 'plot_data/traj_default_start.npy')
traj_robot_path_2   = os.path.join('./results/robot_1', exp_name, 'plot_data/traj_under_start.npy')

path_density = np.load(traj_density_path).astype(np.double)
path_robot   = np.load(traj_robot_path).astype(np.double)
path_robot_2   = np.load(traj_robot_path_2).astype(np.double)

traj_robot = RobotTrajectory(robot, range(len(path_robot)), path_robot) # path: np.array, K x d 
traj_robot_2 = RobotTrajectory(robot, range(len(path_robot_2)), path_robot_2) # path: np.array, K x d 

# resource.edit("Planned trajectory", traj_robot, world=world)

vis.add("world",world)
vis.animate(("world", robot.getName()), path_robot)
vis.add("trajectory_above", traj_robot)   #this draws the end effector trajectory

vis.add("world2",world)
vis.animate(("world2", robot2.getName()), path_robot_2)
vis.add("trajectory_below", traj_robot_2)   #this draws the end effector trajectory

vis.spin(float('inf'))


# print("Planner creation time",time.time() - t0)
# t0 = time.time()
# print("Planning...")
# # n_plan = 1
# n_plan = 10
# numIters = 0
# for round in range(n_plan):
#     planner.planMore(500)
#     numIters += 1
#     if planner.getPath() is not None:
#       break
# print("Planning time,",numIters,"iterations",time.time()-t0)

# path = planner.getPath()
# if path is not None:
#     print("Got a path with",len(path),"milestones")
# else:
#     print("No feasible path was found")


# ### visualization ###

# #provide some debugging information
# V,E = planner.getRoadmap()
# print(len(V),"feasible milestones sampled,",len(E),"edges connected")

# print("CSpace stats:")
# spacestats = space.getStats()
# for k in sorted(spacestats.keys()):
#     print(" ",k,":",spacestats[k])

# print("Planner stats:")
# planstats = planner.getStats()
# for k in sorted(planstats.keys()):
#     print(" ",k,":",planstats[k])

# if path:
#     # save planned milestone path to disk
#     print("Saving to my_plan.configs")
#     resource.set("my_plan.configs",path,"Configs")

#     # visualize path as a Trajectory resource
#     traj = RobotTrajectory(robot,range(len(path)), path) # path: np.array, K x d 
#     resource.edit("Planned trajectory", traj, world=world)

#     #Here's another way to do it: visualize path in the vis module
#     from klampt import vis
#     vis.add("world",world)
#     vis.animate(("world",robot.getName()), path)
#     vis.add("trajectory", traj)   #this draws the end effector trajectory
#     vis.spin(float('inf'))


