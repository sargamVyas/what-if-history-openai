#Problem Statement 
# Imagining how the world will look like if the mentioned historical event hasn't happened
# I have created 3 agents for it - Historian Agent, Alternate Reality Agent, Story teller agent


import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

#Loading environment variables
load_dotenv()

#Import openai API key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

serper_api_key = os.getenv("SERPER_API_KEY")

if not serper_api_key:
    st.error("Serper API key is missing. Please add it to the .env file.")
    st.stop()

#Ensure API key exists
if not openai_api_key:
    st.error("OpenAI API key is missing. Please add it to the .env file.")
    st.stop()

#Streamlit Page config
st.set_page_config(page_title='Reimagined Historical Event', layout='wide')

#Set title and description
st.title('Reimagined Historical Event')
st.markdown('Generate comprehensive blog post about how world will look like if historical event has not happened')

# Sidebar
with st.sidebar:
    st.header('Content Settings')

    topic = st.text_area(
        "Enter historical event",
        height = 100,
        placeholder = 'Enter historical event which you would like to reimagine '
    )

    #More sidebar control
    st.markdown('Temperature Settings')
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)


    #Adding some space
    st.markdown("---")

    generate_button = st.button("Generate Content", type="primary", use_container_width=True)
    
    # Add some helpful information
    with st.expander("How to use"):
        st.markdown("""
        1. Enter your desired topic in the text area above
        2. Adjust the temperature if needed (higher = more creative)
        3. Click 'Generate Content' to start
        4. Wait for the AI to generate your article
        5. Download the result as a markdown file
        """)

    def generate_content(topic):

        llm = LLM(
            model = "gpt-3.5-turbo",
            temperature=0.7,
            api_key=openai_api_key
        )

        search_tool = SerperDevTool(api_key=serper_api_key, n_results=10)

        # First Agent: Historian Agent
        historian_agent = Agent(
            role = "historian_agent",
            goal = f"Research and provide factual, well-documented historical accounts related to {topic}.",
                backstory="You're a highly skilled historian with deep knowledge of world history, key events, and their impacts. "
              "You specialize in fact-based research, providing verified historical information with citations. "
              "Your expertise includes analyzing historical records, primary sources, and academic research "
              "to ensure accuracy and context. You deliver well-structured summaries, ensuring that all facts "
              "are cross-referenced from reliable sources.",
                allow_delegation=False,
        verbose=True,
        tools=[search_tool],  # A tool for scenario generation, logic-based reasoning
        llm=llm
        )

        #Alternate Reality Generator Agent
        alternate_reality_agent = Agent(
        role="Alternate Reality Generator Agent",
        goal=f"Analyze and generate plausible alternate history scenarios based on the modification of {topic}.",
        backstory="You're an expert in speculative history and counterfactual analysis. "
                "You excel at logical reasoning, identifying key turning points in history, "
                "and predicting the cascading effects of alternative outcomes. "
                "You analyze potential political, social, and technological impacts of historical changes "
                "to create coherent, well-reasoned alternate realities.",
        allow_delegation=False,
        verbose=True,
        llm=llm
        )

        #Storyteller agent

        storyteller_agent = Agent(
        role="Storyteller Agent",
        goal=f"Craft immersive and engaging alternative history narratives based on the scenario of {topic}.",
        backstory="You're a talented writer and storyteller specializing in historical fiction and alternate history. "
                  "You excel at translating complex historical and speculative ideas into compelling narratives. "
                  "Your expertise includes character development, world-building, and dramatic storytelling. "
                  "You adapt your writing style to match the intended audience, making history feel alive.",
        allow_delegation=False,
        verbose=True,
        llm=llm
        )
    

        historical_research_task = Task(
            description=("""
                1. Conduct in-depth research on {topic}, including:
                    - Key historical events, dates, and figures
                    - Political, social, and economic impacts
                    - Primary sources, documents, and verified historical records
                2. Cross-reference sources for accuracy and fact-check all information
                3. Structure findings into a well-organized historical research brief
                4. Provide citations and links to original sources for verification
            """),
            expected_output="""A structured historical research brief containing:
                - A concise summary of the event with key details
                - Analyzed impacts on politics, society, and technology
                - Verified sources with citations and references
                - Timeline of significant developments
                - Bullet-pointed key insights for easy reference.""",
                agent=historian_agent
            )

        alternate_history_task = Task(
             description=("""
                Based on the historical research brief, generate a plausible alternative history scenario by:
                1. Identifying key decision points where history could have taken a different turn
                2. Exploring logical consequences of the altered event, including:
                    - Political, economic, and technological changes
                    - Social and cultural shifts
                    - Possible long-term ripple effects
                3. Ensuring the scenario remains consistent with known historical dynamics
                4. Structuring the output to be clear and logically connected
            """),
            expected_output="""A well-structured alternate history analysis containing:
                - A description of the altered event and why it changed
                - A cause-and-effect analysis of immediate consequences
                - A long-term projection of potential global impact
                - Multiple potential outcomes based on historical logic
                - A final summary comparing reality vs. the alternate timeline.""",
                agent=alternate_reality_agent
            )


        storytelling_task = Task(
            description=("""
                Using the alternate history scenario, craft an immersive narrative that:
                1. Presents the alternate timeline in a compelling storytelling format
                2. Includes:
                    - A gripping introduction to set the stage
                    - Key historical characters and their new roles in the altered world
                    - A well-paced narrative structure with dramatic tension
                3. Adapts the tone based on the genre (e.g., documentary, dystopian, sci-fi)
                4. Concludes with thought-provoking insights on how this change could have shaped our world
            """),
            expected_output="""A captivating alternate history story that:
                - Reads like a historical fiction piece or documentary script
                - Engages the reader with vivid descriptions and character perspectives
                - Uses a structured plot with a beginning, middle, and end
                - Reflects realistic consequences of the alternate timeline
                - Is formatted in markdown with clear sections and headings""",
            agent=storyteller_agent
            )

    # Create Crew
        crew = Crew(
            agents=[historian_agent, alternate_reality_agent, storyteller_agent],
            tasks=[historical_research_task, alternate_history_task, storytelling_task],
            verbose=True
        )

        # Kick off the workflow with a given topic
        return crew.kickoff(inputs={"topic": topic})


# Main content area
if generate_button:
    with st.spinner('Generating content... This may take a moment.'):
        try:
            result = generate_content(topic)
            st.markdown("### Generated Content")
            st.markdown(result)
            
            # Add download button
            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"{topic.lower().replace(' ', '_')}_article.md", 
		mime="text/markdown",
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")