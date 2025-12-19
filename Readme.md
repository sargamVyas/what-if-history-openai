# 🏛️ Reimagined Historical Event Generator

An AI-powered application that explores "What If?" scenarios in world history. Using **CrewAI** and **Streamlit**, this tool orchestrates three specialized autonomous agents to research a historical event, analyze a pivot point, and craft a compelling narrative of an alternate reality.

## 🤖 The Crew
The system uses a sequential workflow where each agent builds upon the previous one's output:

1.  **Historian Agent**: Researches factual historical accounts and cross-references primary sources to ground the story in reality.
2.  **Alternate Reality Agent**: Identifies "turning points" and predicts the cascading political, social, and technological effects of a change.
3.  **Storyteller Agent**: Translates complex speculative data into an immersive, high-quality narrative or documentary-style script.



## 🚀 Features
- **Multi-Agent Orchestration**: Powered by CrewAI for complex task delegation.
- **Web Search Integration**: Uses SerperDevTool for real-time historical verification.
- **Customizable Creativity**: Adjust "Temperature" settings via the sidebar to control how wild the alternate history becomes.
- **Exportable Content**: Download your generated alternate history as a formatted Markdown file.

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/historical-reimaginator.git](https://github.com/your-username/historical-reimaginator.git)
cd historical-reimaginator