from swarms import SwarmRouter
from swarms import AutoSwarm
from my_swarm import MySwarm

router = SwarmRouter(swarms=[MySwarm])


# Initialize
autoswarm = AutoSwarm(
    name="kyegomez/myswarm",
    description="A simple API to build and run swarms",
    verbose=True,
    router=router,
)

autoswarm.run("Analyze these financial data and give me a summary")
