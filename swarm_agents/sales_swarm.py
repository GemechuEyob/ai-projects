from swarms import BaseSwarm, Agent, Anthropic


class SalesSwarm(BaseSwarm):
    def __init__(self, name="kyegomez/salesswarm", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        # Define and add agents
        self.customer_data_analyzer = Agent(
            agent_name="Customer Data Analyzer",
            system_prompt="Analyze customer data to identify opportunities.",
            llm=Anthropic(),
            max_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )
        self.pitch_generator = Agent(
            agent_name="Pitch Generator",
            system_prompt="Generate personalized sales pitches based on customer data.",
            llm=Anthropic(),
            max_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )
        self.revenue_forecaster = Agent(
            agent_name="Revenue Forecaster",
            system_prompt="Forecast revenue based on sales pitches and customer data.",
            llm=Anthropic(),
            max_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )

    def run(self, task: str, *args, **kwargs):
        # Analyze customer data
        analyzed_data = self.customer_data_analyzer.run(task, *args, **kwargs)
        # Generate personalized sales pitch
        pitch = self.pitch_generator.run(analyzed_data, *args, **kwargs)
        # Forecast revenue based on pitch and customer data
        revenue_forecast = self.revenue_forecaster.run(pitch, *args, **kwargs)
        return revenue_forecast


SalesSwarm().run("Analyze these financial data and give me a summary")
