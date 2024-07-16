import rvo2
import matplotlib.pyplot as plt

""" param
    --- timeStep: The time step of the simulation
    --- neighborDist: observation radius
    --- maxNeighbors: The maximal number of other agents the agent takes into account in the navigation.
    --- timeHorizon: 如果 timeHorizon 设置为 5，则当前代理在计算新速度时，会考虑在未来 5 秒内与其他代理发生碰撞的可能性.
                     值越大，当前代理考虑的时间范围就越长，代理越早做出反应(lead to moderate speed change)。with respect to other agents.
    --- timeHorizonObst: with respect to obstacles. be same with the last one.
    --- radius: radius of the USV/agent body.
    --- maxSpeed: max speed = (vx**2 + vy**2)**0.5
"""

sim = rvo2.PyRVOSimulator(1/10., 1.5, 5, 1.5, 2, 0.4, 2)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((0, 0))
a1 = sim.addAgent((1, 0))
a2 = sim.addAgent((1, 1))
#                init pos                        now speed
a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))

# Obstacles are also supported.
o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
sim.processObstacles()

sim.setAgentPrefVelocity(a0, (1, 1))
sim.setAgentPrefVelocity(a1, (-1, 1))
sim.setAgentPrefVelocity(a2, (-1, -1))
sim.setAgentPrefVelocity(a3, (1, -1))

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

# Create a list to store the positions of each agent at each time step
agent_positions = [[] for _ in range(sim.getNumAgents())]

for step in range(200):
    sim.doStep()

    # Store the positions of each agent
    for i in range(sim.getNumAgents()):
        agent_positions[i].append(sim.getAgentPosition(i))

    positions = ['(%5.3f,%5.3f)' % sim.getAgentPosition(agent_no)
                 for agent_no in (a0,a1,a2,a3)]
    print('step=%2i t=%.3f %s' % (step,sim.getGlobalTime(),' '.join(positions)))

# Plot the trajectory of each agent
for i in range(sim.getNumAgents()):
    x_coords = [pos[0] for pos in agent_positions[i]]
    y_coords = [pos[1] for pos in agent_positions[i]]
    plt.plot(x_coords,y_coords,label=f'Agent {i}')

# Plot the obstacle
# x_coords = [vertex[0] for vertex in sim.getObstacleVertex()]
# y_coords = [vertex[1] for vertex in sim.getObstacleVertex()]
# plt.plot(x_coords,y_coords,label='Obstacle',color='black')

plt.legend()
plt.show()
