# EVRP-SPD-BD Optimization

This project addresses the **Electric Vehicle Routing Problem with Simultaneous Pickup and Delivery and Battery Degradation (EVRP-SPD-BD)**. I have developed a multi-objective optimization framework that balances the minimization of total travel distance with the preservation of EV fleet battery health. By mathematically modeling non-linear battery degradation—specifically managing Depth of Discharge (DoD) and charging cycle dynamics under time-window constraints—I generate high-quality, near-optimal routing schedules. These heuristic solutions successfully establish a practical trade-off, maximizing both logistical efficiency and the long-term operational lifespan of the batteries.

## Current Repository Status
* Implemented core routing algorithms supporting Simultaneous Pickup/Delivery and Time Windows constraints.
* Built battery physical modeling penalty logic assessing Depth of Discharge (DoD) impacts.
* Generated performance baselines resolving feasible initial solution states.

*(More updates and GA optimizations to be pushed soon).*
