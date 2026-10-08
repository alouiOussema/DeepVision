from common.schemas import Alert
from agents.monitor.agent import MonitorAgent

import pandas as pd
a = Alert(source="ci", signal="build_failed", score=1.0, threshold=1.0,
          details={"commit": "abc123"})
print(a)
Alert(source="cii", signal="x", score=1, threshold=1)   # should raise an error


agent = MonitorAgent()
agent.check_ci()